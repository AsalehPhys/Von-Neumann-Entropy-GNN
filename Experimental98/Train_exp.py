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
    'processed_dir': './processed_experimental',  
    'processed_file_name': 'data.pt',
    'batch_size': 1024,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'hidden_channels': 512,
    'num_epochs': 500,
    'patience': 75,
    'random_seed': 42,
    'best_model_path': 'best_experimental_gnn_98.pth',
    'dropout_p': 0.4,
    'scheduler_factor': 0.5,
    'scheduler_patience': 10,
    'grad_clip': 1.0,
    'curriculum_alpha': 0.5
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
# Old-Style SpinSystemDataset
# -----------------------------------------------------------
class SpinSystemDataset(InMemoryDataset):
    """
    Loads the data.pt from processed_dir
    (the same style as your old code).
    """
    def __init__(self, root='.', transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

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

class ListDataset(InMemoryDataset):
    """Simple dataset class that wraps a list of data objects."""
    def __init__(self, data_list):
        super().__init__()
        self._data_list = data_list
        
    def __len__(self):  
        return len(self._data_list)
    
    def __getitem__(self, idx): 
        return self._data_list[idx]

# -----------------------------------------------------------
# EnhancedPhysicsGNN
# -----------------------------------------------------------
class ExperimentalGNN(nn.Module):
    """
    GNN using only experimentally accessible quantities.
    Still outputs two values:
      [0]: log(S)/N
      [1]: S/N
    """
    def __init__(
        self,
        num_node_features,
        edge_attr_dim,
        hidden_channels=256,
        num_layers=10,
        dropout_p=0.4
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        # Node embedding (experimental features only)
        self.init_transform = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
        )

        # Message Passing (same architecture)
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

        # Readouts
        self.set2set_readout = Set2Set(hidden_channels, processing_steps=4)
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.global_attention = GlobalAttention(gate_nn=self.gate_nn)

        # Global transform (experimental features only)
        self.global_transform = nn.Sequential(
            nn.Linear(7, hidden_channels),  # Reduced from 10 to 7 experimental features
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Size encoding
        self.size_encoder = nn.Sequential(
            nn.Linear(1, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels // 2)
        )

        # Final MLP - multi-task
        combined_in_dim = (2 * hidden_channels) + hidden_channels + hidden_channels + (hidden_channels // 2)
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_in_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels // 2, 2)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Node embedding
        h = self.init_transform(x)
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index, edge_attr)
            h_new = self.norms[i](h_new)
            h = h + h_new  # Residual

        # Graph-level readouts
        s2s = self.set2set_readout(h, batch)
        ga = self.global_attention(h, batch)

        # Experimental global features only
        system_size = data.system_size.squeeze(-1)
        total_ryd = data.total_rydberg
        dens_ryd = data.rydberg_density
        config_ent = data.config_entropy.squeeze(-1)
        rel_ent = config_ent / torch.log(system_size + 1e-6)
        nA = data.nA.squeeze(-1)
        nB = data.nB.squeeze(-1)

        global_feats = torch.stack([
            (total_ryd / system_size),
            dens_ryd / system_size,
            system_size,
            config_ent / system_size,
            rel_ent,
            nA / system_size,
            nB / system_size,
        ], dim=1)

        gf_out = self.global_transform(global_feats)
        size_encoded = self.size_encoder(system_size.unsqueeze(-1))
        combined = torch.cat([s2s, ga, gf_out, size_encoded], dim=-1)
        out = self.final_mlp(combined)
        return out

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
        """Get bounds on entropy (lower bound is now zero)"""
        # Lower bound is zero
        lower_bound = torch.zeros_like(system_size, dtype=torch.float)
        # Upper bound remains the same
        min_size = torch.minimum(subsystem_size.float(), (system_size - subsystem_size).float())
        upper_bound = min_size * torch.log(torch.tensor(2.0, device=system_size.device))
        return lower_bound, upper_bound

    def forward(self, pred_log_s_over_n, target, system_size, subsystem_size):
        # Convert prediction to actual entropy (S_pred)
        pred_entropy = torch.exp(pred_log_s_over_n * system_size)

        # Physical bounds (no MI)
        lower_bound, upper_bound = self.get_entropy_bounds(system_size, subsystem_size)

        # Out-of-bounds penalty
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

        # MSE on log(S)/N
        log_target = torch.log(target + 1e-10) / system_size
        base_loss = F.mse_loss(pred_log_s_over_n, log_target, reduction='none')

        # Weighting by system size
        size_weight = (system_size.float() / self.base_size) ** self.scaling_power
        weighted_loss = base_loss * size_weight

        return (weighted_loss + self.physics_weight * physics_loss).mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, physics_weight=0.3, alpha_s_over_n=0.2):
        super().__init__()
        self.phys_loss = PhysicalScaleAwareLoss(physics_weight=physics_weight)
        self.alpha_s_over_n = alpha_s_over_n

    def forward(self, preds, targets, system_size, subsystem_size):
        pred_log_s_over_n = preds[:, 0]
        pred_s_over_n = preds[:, 1]

        # Physical scale aware loss without MI bound
        loss1 = self.phys_loss(
            pred_log_s_over_n, 
            targets, 
            system_size, 
            subsystem_size
        )

        # MSE on second head
        true_s_over_n = targets / system_size
        loss2 = F.mse_loss(pred_s_over_n, true_s_over_n)

        return loss1 + self.alpha_s_over_n * loss2
# -----------------------------------------------------------
# Curriculum Sampler
# -----------------------------------------------------------
def get_curriculum_sampler(dataset, epoch, max_epochs, alpha=0.5):
    system_sizes = [data.system_size.item() for data in dataset]
    if len(system_sizes) == 0:
        return None

    max_size = max(system_sizes)
    progress = min(1.0, epoch / (max_epochs * 0.7))

    weights = []
    for size in system_sizes:
        size_ratio = (size - 4) / (max_size - 4 + 1e-6)
        w = 1.0 + (progress ** alpha) * size_ratio
        weights.append(w if w > 0 else 1e-6)

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

        # Forward
        preds = model(data)

        # Exclude system size 16/18 from the loss
        mask = (data.system_size.squeeze(-1) != 16) & (data.system_size.squeeze(-1) != 18)
        if not mask.any():
            continue

        subsystem_size = data.nA.squeeze(-1)[mask]
        system_size_ = data.system_size.squeeze(-1)[mask]
        targets_ = data.y.squeeze()[mask]
        preds_ = preds[mask]

        loss = criterion(preds_, targets_, system_size_, subsystem_size)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device, name='Eval'):
    model.eval()
    total_loss = 0.0
    predictions_abs, predictions_over_n = [], []
    targets, sizes = [], []
    
    for data in loader:
        data = data.to(device)
        preds = model(data)
        
        mask = (data.system_size.squeeze(-1) != 16) & (data.system_size.squeeze(-1) != 18)
        if mask.any():
            subsystem_size = data.nA.squeeze(-1)[mask]
            loss = criterion(
                preds[mask],
                data.y.squeeze()[mask],
                data.system_size.squeeze(-1)[mask],
                subsystem_size
            )
            total_loss += loss.item() * mask.sum().item()

        pred_log_s_over_n = preds[:, 0]
        pred_entropy_abs = torch.exp(pred_log_s_over_n * data.system_size.squeeze(-1))
        
        predictions_abs.append(pred_entropy_abs.cpu())
        predictions_over_n.append(preds[:,1].cpu())
        targets.append(data.y.squeeze().cpu())
        sizes.append(data.system_size.squeeze().cpu())

    # Count total non-[16,18] samples
    total_valid_samples = sum(1 for d in loader.dataset if d.system_size.item() not in [16, 18])
    mean_loss = total_loss / total_valid_samples if total_valid_samples > 0 else 0.0

    # Concatenate tensors
    predictions_abs = torch.cat(predictions_abs)
    predictions_over_n = torch.cat(predictions_over_n)
    targets = torch.cat(targets)
    sizes = torch.cat(sizes)

    # Convert to numpy for metrics
    predictions_abs = predictions_abs.numpy()
    predictions_over_n = predictions_over_n.numpy()
    targets = targets.numpy()
    sizes = sizes.numpy()

    # Compute metrics per size
    metrics_str = []
    size_metrics = {}
    for sz in sorted(np.unique(sizes)):
        sz_int = int(sz)
        mask = (sizes == sz)
        size_preds = predictions_abs[mask]
        size_targets = targets[mask]
        
        mse = mean_squared_error(size_targets, size_preds)
        mae = mean_absolute_error(size_targets, size_preds)
        mape = np.mean(np.abs(size_preds - size_targets) / (size_targets + 1e-10)) * 100
        
        size_metrics[sz_int] = {
            'mse': mse, 
            'mae': mae, 
            'mape': mape
        }
        
        metrics_line = f"Size {sz_int:2}: MSE={mse:.2e} MAE={mae:.2e} MAPE={mape:.1f}%"
        if sz_int in [16, 18]:
            metrics_line += " (test only)"
        metrics_str.append(metrics_line)

    # Print summary
    logging.info(f"\n{name} Summary:")
    if total_valid_samples > 0:
        logging.info(f"Mean Loss (excl. size 16,18): {mean_loss:.6f}")
    logging.info("Size-specific metrics:")
    for line in metrics_str:
        logging.info(f"  {line}")

    return {
        'loss': mean_loss,
        'predictions_abs': predictions_abs,
        'predictions_s_over_n': predictions_over_n,
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

    # Load experimental dataset
    dataset = SpinSystemDataset(root=CONFIG['processed_dir'])
    if len(dataset) == 0:
        logging.error("Loaded dataset is empty. Exiting.")
        return

    # Split data
    dataset_sizes = [dataset[i].system_size.item() for i in range(len(dataset))]
    train_val_subset = [dataset[i] for i in range(len(dataset))]
    test_subset = [
        dataset[i]
        for i in range(len(dataset))
        if dataset_sizes[i] in [16, 18]
    ]

    # Create datasets
    train_val_dataset = ListDataset(train_val_subset)
    test_dataset = ListDataset(test_subset)

    # Train/val split
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )

    # Build model with experimental features
    sample_data = next(iter(DataLoader(train_dataset, batch_size=1)))
    model = ExperimentalGNN(
        num_node_features=sample_data.x.size(1),
        edge_attr_dim=sample_data.edge_attr.size(1),
        hidden_channels=CONFIG['hidden_channels'],
        dropout_p=CONFIG['dropout_p'],
        num_layers=10
    ).to(device)

    criterion = MultiTaskLoss(physics_weight=0.3, alpha_s_over_n=0.2)
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
            sampler = get_curriculum_sampler(
                ds, epoch, CONFIG['num_epochs'], alpha=CONFIG['curriculum_alpha']
            )
        return DataLoader(ds, batch_size=CONFIG['batch_size'], sampler=sampler)

    best_val_loss = float('inf')

    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        logging.info(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        train_loader = create_dataloader(train_dataset, epoch, shuffle=True)
        val_loader = create_dataloader(val_dataset, epoch, shuffle=False)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, clip_grad=CONFIG['grad_clip']
        )
        logging.info(f"  Training Loss (excl. size16/18): {train_loss:.6f}")

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, name='Validation')
        val_loss = val_metrics['loss']
        scheduler.step(val_loss)

        # Check if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['best_model_path'])
            logging.info(f"  [Info] New best model saved (val_loss={best_val_loss:.6f})")

        # Also evaluate on sizes 16 and 18 each epoch (does not affect scheduler or best model)
        if test_loader is not None:
            logging.info("  Logging size16/18 MAPE (not factored into loss/optimization):")
            evaluate(model, test_loader, criterion, device, name='Test16,18-EpochCheck')

    # Final test on sizes 16, 18
    if test_loader is not None:
        logging.info("Reloading best model for final test on sizes [16, 18]...")
        model.load_state_dict(torch.load(CONFIG['best_model_path'], map_location=device))
        _ = evaluate(model, test_loader, criterion, device, name='Final Test (16,18)')
    else:
        logging.warning("No data found with system_size in [16,18]. Skipping final test.")

if __name__ == "__main__":
    main()
