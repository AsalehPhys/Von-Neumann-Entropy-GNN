"""
Train.py

Loads the dataset the old-style way:
    dataset = SpinSystemDataset(root=CONFIG['processed_dir'])
which expects 'data.pt' in that directory.

Then splits by system size, builds an EnhancedPhysicsGNN, and trains with
PhysicalScaleAwareLoss.
"""

import os
import sys
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, DataLoader
from torch.utils.data import random_split, WeightedRandomSampler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from torch_geometric.nn import (
    GINEConv, 
    GlobalAttention, 
    Set2Set, 
    TransformerConv
)
from torch_geometric.nn.norm import BatchNorm

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
CONFIG = {
    'processed_dir': './processed2',  # Directory containing 'data.pt'
    'processed_file_name': 'data.pt',
    'batch_size': 1024,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'hidden_channels': 512,
    'num_epochs': 500,
    'patience': 50,
    'random_seed': 42,
    'best_model_path': 'best_gnn_model.pth',
    'dropout_p': 0.4,
    'scheduler_factor': 0.5,
    'scheduler_patience': 10,
    'grad_clip': 1.0,
    'curriculum_steps': 5
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
# Old-Style SpinSystemDataset (NO dataframe in constructor)
# -----------------------------------------------------------
class SpinSystemDataset(InMemoryDataset):
    """
    Loads the data.pt from processed_dir
    (the same style as your old code).
    """
    def __init__(self, root='.', transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # The line below expects that `data.pt` already exists in `self.processed_paths[0]`
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # No raw files needed if we only load from data.pt
        return []

    @property
    def processed_file_names(self):
        return [CONFIG['processed_file_name']]

    def download(self):
        # Not used; we assume data.pt is already there
        pass

    def process(self):
        # Not used; we assume data.pt is already there
        pass

# -----------------------------------------------------------
# EnhancedPhysicsGNN
# -----------------------------------------------------------
class EnhancedPhysicsGNN(nn.Module):
    """
    GNN with alternating GINEConv/TransformerConv and global readouts.
    Expects 10 global features: [Omega, Delta, Energy/size, RydExc/size, density, size, configEnt, relativeEnt, nA, nB]
    """
    def __init__(
        self,
        num_node_features,
        edge_attr_dim,
        hidden_channels=256,
        num_layers=8,
        dropout_p=0.4
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        # 1) Node embedding
        self.init_transform = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
        )

        # 2) Message Passing
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                mp_mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                conv = GINEConv(mp_mlp, edge_dim=edge_attr_dim)
            else:
                conv = TransformerConv(
                    hidden_channels, hidden_channels // 4,
                    heads=4,
                    edge_dim=edge_attr_dim,
                    dropout=dropout_p,
                    beta=True
                )
            self.convs.append(conv)
            self.norms.append(BatchNorm(hidden_channels))

        # 3) Readouts
        self.set2set_readout = Set2Set(hidden_channels, processing_steps=4)
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.global_attention = GlobalAttention(gate_nn=self.gate_nn)

        # 4) Global Feature Transform (10 -> hidden_channels)
        self.global_transform = nn.Sequential(
            nn.Linear(10, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # 5) Final MLP
        combined_in_dim = (2 * hidden_channels) + hidden_channels + hidden_channels
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_in_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Node-level embed
        h = self.init_transform(x)
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index, edge_attr)
            h_new = self.norms[i](h_new)
            h = h + h_new  # Residual

        s2s = self.set2set_readout(h, batch)
        ga = self.global_attention(h, batch)

        # 10 global features
        Omega = data.Omega.squeeze(-1)
        Delta = data.Delta.squeeze(-1)
        Energy = data.Energy.squeeze(-1)
        system_size = data.system_size.squeeze(-1)
        total_ryd = data.total_rydberg
        dens_ryd = data.rydberg_density
        config_ent = data.config_entropy.squeeze(-1)
        rel_ent = config_ent / torch.log(system_size + 1e-6)

        nA = data.nA.squeeze(-1)
        nB = data.nB.squeeze(-1)

        global_feats = torch.stack([
            Omega,
            Delta,
            Energy / system_size,
            (total_ryd / system_size),
            dens_ryd,
            system_size,
            config_ent,
            rel_ent,
            nA,
            nB,
        ], dim=1)

        gf_out = self.global_transform(global_feats)
        combined = torch.cat([s2s, ga, gf_out], dim=-1)
        out = self.final_mlp(combined)
        return out.squeeze(-1)

# -----------------------------------------------------------
# PhysicalScaleAwareLoss
# -----------------------------------------------------------
class PhysicalScaleAwareLoss(nn.Module):
    def __init__(self, base_size=4, scaling_power=1.5, physics_weight=1.0):
        super().__init__()
        self.base_size = base_size
        self.scaling_power = scaling_power
        self.physics_weight = physics_weight

    def get_entropy_bounds(self, system_size, subsystem_size):
        lower_bound = torch.zeros_like(system_size, dtype=torch.float)
        min_size = torch.minimum(subsystem_size.float(), (system_size - subsystem_size).float())
        upper_bound = min_size * torch.log(torch.tensor(2.0, device=system_size.device))
        return lower_bound, upper_bound

    def forward(self, pred, target, system_size, subsystem_size):
        # 'pred' is log(S_pred)
        if system_size.dim() == 2:
            system_size = system_size.squeeze(-1)
        if subsystem_size.dim() == 2:
            subsystem_size = subsystem_size.squeeze(-1)

        pred_entropy = torch.exp(pred)
        lower_bound, upper_bound = self.get_entropy_bounds(system_size, subsystem_size)

        # out-of-bounds penalty
        lower_violation = F.smooth_l1_loss(
            torch.maximum(lower_bound, pred_entropy),
            pred_entropy,
            reduction='none'
        )
        upper_violation = F.smooth_l1_loss(
            torch.minimum(upper_bound, pred_entropy),
            pred_entropy,
            reduction='none'
        )
        physics_loss = lower_violation + upper_violation

        # base MSE on log(target)
        log_target = torch.log(target + 1e-10)
        base_loss = F.mse_loss(pred, log_target, reduction='none')

        # weighting by system size
        size_weight = (system_size.float() / self.base_size) ** self.scaling_power
        weighted_loss = base_loss * size_weight

        return (weighted_loss + self.physics_weight * physics_loss).mean()

# -----------------------------------------------------------
# Curriculum
# -----------------------------------------------------------
def get_curriculum_sampler(dataset, epoch, max_epochs):
    system_sizes = [data.system_size.item() for data in dataset]
    max_size = max(system_sizes) if len(system_sizes) > 0 else 1
    progress = min(1.0, epoch / (max_epochs * 0.7))

    weights = []
    for size in system_sizes:
        if size < 8:
            weight = 1.0
        else:
            weight = progress * (1.0 - (size - 8)/(max_size - 8 + 1e-6))
            weight = max(0.1, weight)
        weights.append(weight)

    return WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)

# -----------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device, clip_grad=None):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        # use data.nA as subsystem_size
        subsystem_size = data.nA.squeeze(-1)
        loss = criterion(out, data.y.squeeze(), data.system_size, subsystem_size)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device, name='Eval'):
    model.eval()
    total_loss = 0.0
    predictions, targets, sizes = [], [], []

    for data in loader:
        data = data.to(device)
        out = model(data)

        subsystem_size = data.nA.squeeze(-1)
        loss = criterion(out, data.y.squeeze(), data.system_size, subsystem_size)
        total_loss += loss.item() * data.num_graphs

        predictions.append(torch.exp(out).cpu())
        targets.append(data.y.squeeze().cpu())
        sizes.append(data.system_size.squeeze().cpu())

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    sizes = torch.cat(sizes).numpy()

    unique_sizes = np.unique(sizes)
    size_metrics = {}
    for sz in unique_sizes:
        mask = (sizes == sz)
        size_preds = predictions[mask]
        size_targets = targets[mask]
        size_metrics[int(sz)] = {
            'mse': mean_squared_error(size_targets, size_preds),
            'mae': mean_absolute_error(size_targets, size_preds),
            'mape': np.mean(np.abs(size_preds - size_targets) / (size_targets + 1e-10)) * 100
        }

    mean_loss = total_loss / len(loader.dataset) if len(loader.dataset) else 0.0

    logging.info(f"[{name}] Loss: {mean_loss:.6f}")
    for sz, met in size_metrics.items():
        logging.info(
            f"  Size={sz:2d} : "
            f"MSE={met['mse']:.6f} "
            f"MAE={met['mae']:.6f} "
            f"MAPE={met['mape']:.2f}%"
        )

    return {
        'loss': mean_loss,
        'predictions': predictions,
        'targets': targets,
        'sizes': sizes,
        'size_metrics': size_metrics
    }

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    setup_logging()
    set_seed(CONFIG['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("Loading dataset the old way (SpinSystemDataset).")
    dataset = SpinSystemDataset(root=CONFIG['processed_dir'])

    if len(dataset) == 0:
        logging.error("Loaded dataset is empty. Exiting.")
        return

    # Filter by system size
    dataset_sizes = [dataset[i].system_size.item() for i in range(len(dataset))]
    train_val_subset = [dataset[i] for i in range(len(dataset)) if dataset_sizes[i] <= 12]
    size14_subset = [dataset[i] for i in range(len(dataset)) if abs(dataset_sizes[i] - 14) < 1e-6]

    # Convert to list-based
    class ListDataset(InMemoryDataset):
        def __init__(self, data_list):
            self._data_list = data_list
        def __len__(self):
            return len(self._data_list)
        def __getitem__(self, idx):
            return self._data_list[idx]

    train_val_dataset = ListDataset(train_val_subset)
    size14_dataset = ListDataset(size14_subset)

    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    if train_size == 0:
        logging.error("No data found with system_size <= 12.")
        return

    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )

    # Build model
    sample_data = next(iter(DataLoader(train_dataset, batch_size=1)))
    model = EnhancedPhysicsGNN(
        num_node_features=sample_data.x.size(1),
        edge_attr_dim=sample_data.edge_attr.size(1),
        hidden_channels=CONFIG['hidden_channels'],
        dropout_p=CONFIG['dropout_p']
    ).to(device)

    criterion = PhysicalScaleAwareLoss(physics_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['scheduler_factor'],
        patience=CONFIG['scheduler_patience']
    )

    def create_dataloader(ds, epoch, shuffle=False):
        sampler = None
        if shuffle:
            sampler = get_curriculum_sampler(ds, epoch, CONFIG['num_epochs'])
        return DataLoader(ds, batch_size=CONFIG['batch_size'], sampler=sampler)

    best_val_loss = float('inf')

    # Possibly create loader for size=14
    size14_loader = None
    if len(size14_dataset) > 0:
        size14_loader = DataLoader(size14_dataset, batch_size=CONFIG['batch_size'])

    for epoch in range(CONFIG['num_epochs']):
        logging.info(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        train_loader = create_dataloader(train_dataset, epoch, shuffle=True)
        val_loader = create_dataloader(val_dataset, epoch, shuffle=False)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, clip_grad=CONFIG['grad_clip']
        )
        logging.info(f"  Training Loss: {train_loss:.6f}")

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, name='Validation')
        val_loss = val_metrics['loss']

        # Scheduler
        scheduler.step(val_loss)

        # Check if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['best_model_path'])
            logging.info(f"  [Info] New best model saved (val_loss={best_val_loss:.6f})")

        # Optional: check size-14 each epoch
        if size14_loader is not None:
            size14_metrics = evaluate(model, size14_loader, criterion, device, name='Size14-DuringTraining')
            if 14 in size14_metrics['size_metrics']:
                mape_14 = size14_metrics['size_metrics'][14]['mape']
                logging.info(f"  [Size14 MAPE at epoch {epoch+1}]: {mape_14:.2f}%")

    # Final test on size=14
    if size14_loader is not None:
        logging.info("Reloading best model for final size14 test...")
        model.load_state_dict(torch.load(CONFIG['best_model_path'], map_location=device))
        _ = evaluate(model, size14_loader, criterion, device, name='Final Size14-Test')
    else:
        logging.warning("No data found with system_size=14. Skipping final test.")


if __name__ == "__main__":
    main()
