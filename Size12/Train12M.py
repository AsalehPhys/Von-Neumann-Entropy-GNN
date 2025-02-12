import os
import sys
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.utils import dropout_adj
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import (
    GINEConv,
    TransformerConv,
    Set2Set,
    BatchNorm,
    global_add_pool,
    LayerNorm,
)
from torch.utils.data import random_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.nn import (
    GINEConv, TransformerConv, Set2Set, LayerNorm, global_add_pool, GraphNorm
)
from torch_geometric.utils import dropout_adj

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
CONFIG = {
    'processed_dir': './processed_experimental12',
    'processed_file_name': 'data.pt',
    'batch_size': 1024,
    'learning_rate': 4e-4,
    'weight_decay': 1e-4,
    'hidden_channels': 768,
    'num_epochs': 500,
    'patience': 75,
    'random_seed': 42,
    'best_model_path': 'best_model.pth',
    'dropout_p': 0.4,
    'scheduler_factor': 0.5,
    'scheduler_patience': 10,
    'grad_clip': 1.0,
}

# -----------------------------------------------------------
# Logging & Utilities
# -----------------------------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# -----------------------------------------------------------
# SpinSystemDataset
# -----------------------------------------------------------
class SpinSystemDataset(InMemoryDataset):
    """Loads the data.pt from processed_dir (old-style)."""
    def __init__(self, root='.', transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [CONFIG['processed_file_name']]

    def download(self):
        pass

    def process(self):
        pass

# -----------------------------------------------------------
# PhysicalScaleAwareLoss (Single-Head)
# -----------------------------------------------------------
class PhysicalScaleAwareLoss(nn.Module):
    """
    A combined loss that balances:
      1) Relative error (to reduce percentage error).
      2) A small MSE or log-cosh term for stability.
      3) Physics-based bounds as soft constraints.
    """
    def __init__(self, physics_weight=1.0, rel_weight=0.5, eps=1e-10):
        super().__init__()
        self.physics_weight = physics_weight
        self.rel_weight = rel_weight
        self.eps = eps

    def get_entropy_bounds(self, system_size, subsystem_size):
        lower_bound = torch.zeros_like(system_size, dtype=torch.float)
        min_size = torch.minimum(subsystem_size.float(), 
                                 (system_size - subsystem_size).float())
        upper_bound = min_size * torch.log(torch.tensor(2.0, 
                                             device=system_size.device))
        return lower_bound, upper_bound

    def forward(self, pred_s, target_s, system_size, subsystem_size):
        # 1. Base MSE/log-cosh for stability
        error = pred_s - target_s
        logcosh = torch.log(torch.cosh(error + self.eps))
        logcosh_loss = logcosh.mean()

        # 2. Relative loss for percentage error
        #    - You can also use a 'smooth L1 style' with something like
        #      relative_loss = F.l1_loss(pred_s / (target_s + eps), 
        #                                torch.ones_like(target_s))
        rel_error = torch.abs(error) / (torch.abs(target_s) + self.eps)
        rel_loss = rel_error.mean()

        # 3. Soft physics-based constraints
        lower_bound, upper_bound = self.get_entropy_bounds(system_size, subsystem_size)
        bounds_violation = (
            torch.relu(lower_bound - pred_s) +
            torch.relu(pred_s - upper_bound)
        ).mean()

        # Combine them:
        # - Weighted sum of log-cosh and relative
        # - Weighted physics-based penalty
        total_loss = (1 - self.rel_weight) * logcosh_loss + \
                     self.rel_weight * rel_loss + \
                     self.physics_weight * bounds_violation

        return total_loss

# -----------------------------------------------------------
# GNN Model (Single Head: log(S/N)) + Global Features
# -----------------------------------------------------------
class ExperimentalGNN(nn.Module):
    def __init__(self, hidden_channels=64, num_layers=10, dropout_p=0.4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.feature_indices = {
            'position': slice(0, 2),
            'rydberg_val': 2,
            'mask': 3,
        }

        # Node and Edge Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(4, hidden_channels),
            LayerNorm(hidden_channels),
            nn.SiLU()
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_channels),
            LayerNorm(hidden_channels),
            nn.SiLU()
        )

        # Hierarchical Convolution Layers
        self.convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                # GINEConv with enhanced edge features
                mp_mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    LayerNorm(hidden_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                conv = GINEConv(mp_mlp, edge_dim=hidden_channels)
            else:
                # TransformerConv with enhanced attention
                conv = TransformerConv(
                    hidden_channels,
                    hidden_channels // 8,
                    heads=8,
                    edge_dim=hidden_channels,
                    dropout=dropout_p,
                    beta=True,
                    concat=True
                )
            self.convs.append(conv)
            self.edge_convs.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            ))
            self.norms.append(LayerNorm(hidden_channels))

        # Hierarchical Pooling
        self.pool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            ) for _ in range(num_layers // 2)
        ])

        # Reduce number of readout heads and add dimension reduction
        self.readout = nn.ModuleList([
            Set2Set(hidden_channels, processing_steps=4) for _ in range(2)  # Reduced from 4 to 2 heads
        ])
        
        # Dimension reduction for readout outputs
        self.readout_projection = nn.Sequential(
            nn.Linear(4 * hidden_channels, 2 * hidden_channels),  # 4 = 2 heads * 2 (Set2Set doubles dim)
            LayerNorm(2 * hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p)
        )

        # Global Features MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Final MLP with corrected dimensions
        combined_dim = (2 * hidden_channels) + hidden_channels  # projected readout + global features
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels),
            LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels // 2),
            LayerNorm(hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Node and Edge Feature Encoding
        node_features = torch.cat([
            x[:, self.feature_indices['position']],
            x[:, self.feature_indices['rydberg_val']].unsqueeze(-1),
            x[:, self.feature_indices['mask']].unsqueeze(-1),
        ], dim=1)

        edge_features = torch.stack([
            edge_attr[:, 0],
            edge_attr[:, 1],
            edge_attr[:, 2]
        ], dim=1)

        x_enc = self.node_encoder(node_features)
        e_enc = self.edge_encoder(edge_features)

        # Message Passing with Enhanced Edge Features
        h = x_enc
        for i in range(self.num_layers):
            # Update edge features
            e_enc = self.edge_convs[i](e_enc)
            # Perform convolution
            h_new = self.convs[i](h, edge_index, e_enc)
            h_new = self.norms[i](h_new)
            h = h + h_new  # Residual connection

            # Hierarchical Pooling with safety check
            if i % 2 == 0 and i // 2 < len(self.pool):
                h = self.pool[i // 2](h)

        # Multi-Head Readout with dimension reduction
        readouts = [readout(h, batch) for readout in self.readout]
        h_readout = torch.cat(readouts, dim=1)
        h_readout = self.readout_projection(h_readout)

        # Global Features
        nA_over_N = data.nA.squeeze(-1) / (data.system_size.squeeze(-1) + 1e-10)
        nB_over_N = data.nB.squeeze(-1) / (data.system_size.squeeze(-1) + 1e-10)
        global_feats = torch.stack([nA_over_N, nB_over_N], dim=1)
        gf_out = self.global_mlp(global_feats)

        # Combine features with correct dimensions
        combined = torch.cat([h_readout, gf_out], dim=1)
        out = self.final_mlp(combined).squeeze(-1)
        return out

# -----------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device, rank, clip_grad=None):
    model.train()
    total_loss = torch.tensor(0.0, device=device)  # Use a tensor for total_loss
    n_samples = torch.tensor(0, device=device)     # Use a tensor for n_samples

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        pred_s= model(data)
        targets = data.y.squeeze()
        system_size = data.system_size.squeeze(-1)
        subsystem_size = data.nA.squeeze(-1)

        loss = criterion(pred_s, targets, system_size, subsystem_size)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        batch_size = torch.tensor(data.num_graphs, device=device)  # Use a tensor for batch_size
        n_samples += batch_size
        total_loss += loss.item() * batch_size

    # Average loss across all processes
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
    return (total_loss / n_samples).item() if n_samples > 0 else 0.0

@torch.no_grad()
def evaluate(model, loader, criterion, device, rank, name='Eval'):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)  # Use a tensor for total_loss
    n_samples = torch.tensor(0, device=device)     # Use a tensor for n_samples

    all_preds_abs = []
    all_targets = []

    for data in loader:
        data = data.to(device)
        pred_s = model(data)

        targets = data.y.squeeze()
        system_size = data.system_size.squeeze(-1)
        subsystem_size = data.nA.squeeze(-1)

        loss = criterion(pred_s, targets, system_size, subsystem_size)
        batch_size = torch.tensor(data.num_graphs, device=device)  # Use a tensor for batch_size
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        all_preds_abs.append(pred_s.cpu())
        all_targets.append(targets.cpu())

    # Gather results from all processes
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
    mean_loss = (total_loss / n_samples).item() if n_samples > 0 else 0.0

    if rank == 0:
        all_preds_abs = torch.cat(all_preds_abs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        mse_abs = mean_squared_error(all_targets, all_preds_abs)
        mae_abs = mean_absolute_error(all_targets, all_preds_abs)
        mape_abs = np.mean(np.abs((all_preds_abs - all_targets) / (all_targets + 1e-10))) * 100

        logging.info(f"\n{name} Summary:")
        logging.info(f"  Loss: {mean_loss:.6f}")
        logging.info(f"  MSE (Absolute S): {mse_abs:.6f}")
        logging.info(f"  MAE (Absolute S): {mae_abs:.6f}")
        logging.info(f"  MAPE (Absolute S): {mape_abs:.2f}%")

    return mean_loss

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main(rank, world_size):
    setup_logging()
    set_seed(CONFIG['random_seed'])

    # Initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Load dataset
    dataset = SpinSystemDataset(root=CONFIG['processed_dir'])
    if len(dataset) == 0:
        logging.error("Loaded dataset is empty. Exiting.")
        return

    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )

    # Create DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], sampler=val_sampler)

    # Initialize the model
    model = ExperimentalGNN(
        hidden_channels=CONFIG['hidden_channels'],
        dropout_p=CONFIG['dropout_p']
    ).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = PhysicalScaleAwareLoss(physics_weight=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['scheduler_factor'],
        patience=CONFIG['scheduler_patience']
    )

    best_val_loss = float('inf')

    for epoch in range(CONFIG['num_epochs']):
        if rank == 0:
            logging.info(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, rank, clip_grad=CONFIG['grad_clip'])

        if rank == 0:
            logging.info(f"  Training Loss: {train_loss:.6f}")

        val_loss = evaluate(model, val_loader, criterion, device, rank, name='Validation')
        scheduler.step(val_loss)

        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.module.state_dict(), CONFIG['best_model_path'])
            logging.info(f"  [Info] Best model saved (val_loss={best_val_loss:.6f})")

    if rank == 0:
        logging.info("Training complete. Loading best model for final validation...")
        model.module.load_state_dict(torch.load(CONFIG['best_model_path'], map_location=device))
        _ = evaluate(model, val_loader, criterion, device, rank, name='Final Validation')

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)