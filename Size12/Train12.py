import sys
import logging
import random
from torch_geometric.utils import dropout_adj
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import (
    GINEConv,
    TransformerConv,
    Set2Set,
    BatchNorm,
    GATConv, 
    global_add_pool, 
    LayerNorm
)
from torch.utils.data import random_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
CONFIG = {
    'processed_dir': './processed_experimental12',
    'processed_file_name': 'data.pt',
    'batch_size': 2048,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'hidden_channels': 512,
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
    1) Predicts log(S/N).
    2) Applies physical bounding penalty for 0 <= S <= min(A, B) * log(2).
    3) Uses size-weighted MSE on log(S/N).
    """
    def __init__(self, base_size=4, scaling_power=1.5, physics_weight=1.0):
        """
        Args:
          base_size: Reference system size for weighting
          scaling_power: Exponent for how strongly bigger systems are weighted
          physics_weight: How heavily to penalize out-of-bounds predictions
        """
        super().__init__()
        self.base_size = base_size
        self.scaling_power = scaling_power
        self.physics_weight = physics_weight

    def get_entropy_bounds(self, system_size, subsystem_size):
        """
        Lower bound = 0
        Upper bound = min(subsystem_size, system_size - subsystem_size) * ln(2)
        """
        lower_bound = torch.zeros_like(system_size, dtype=torch.float)
        min_size = torch.minimum(subsystem_size.float(), (system_size - subsystem_size).float())
        upper_bound = min_size * torch.log(torch.tensor(2.0, device=system_size.device))
        return lower_bound, upper_bound

    def forward(self, pred_log_s_over_n, target_s, system_size, subsystem_size):
        """
        pred_log_s_over_n: shape [batch]
        target_s: shape [batch], the true absolute entropies
        system_size, subsystem_size: shapes [batch]
        """
        # Convert log(S/N) -> S_pred
        pred_entropy = torch.exp(pred_log_s_over_n) * system_size

        # Physical bounds
        lower_bound, upper_bound = self.get_entropy_bounds(system_size, subsystem_size)

        # Smooth L1 penalty if pred_entropy < 0 or > upper bound
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

        # MSE on log(S/N)
        # actual log(S/N) = log(target_s + small_eps) / system_size
        log_target = torch.log((target_s + 1e-10)/system_size) 
        base_loss = F.mse_loss(pred_log_s_over_n, log_target, reduction='none')

        # System-size-based weighting
        size_weight = (system_size.float() / self.base_size) ** self.scaling_power
        weighted_loss = base_loss * size_weight

        total_loss = weighted_loss + self.physics_weight * physics_loss
        return total_loss.mean()

# -----------------------------------------------------------
# GNN Model (Single Head: log(S/N)) + Global Features
# -----------------------------------------------------------
class ExperimentalGNN(nn.Module):
    def __init__(
        self,
        hidden_channels=512,
        num_layers=8,
        dropout_p=0.4,
        num_heads=8,
        node_input_dim=4,  # Adjust if adding more node features
        edge_input_dim=2   # Adjust if adding more edge features
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.num_heads = num_heads

        self.feature_indices = {
            'position': slice(0, 2),  # x, y coordinates
            'rydberg_val': 2,         # Rydberg probability
            'mask': 3,                # Subsystem mask
        }

        # Node feature transformation
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_channels),
            LayerNorm(hidden_channels),
            nn.ELU()
        )

        # Edge feature transformation
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_channels // 2),
            LayerNorm(hidden_channels // 2),
            nn.ELU()
        )

        # Message Passing
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            conv = GATConv(
                hidden_channels,
                hidden_channels // num_heads,
                heads=num_heads,
                dropout=dropout_p,
                edge_dim=hidden_channels // 2
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(hidden_channels))

        # Post-convolution pooling
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout_p),
        )

        # MLP for the global features (nA_over_N, nB_over_N)
        self.global_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels // 2),           # 2 is for [nA/N, nB/N]
            LayerNorm(hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels // 2, hidden_channels // 2),
            LayerNorm(hidden_channels // 2),
            nn.ELU()
        )

        # Final MLP for combining readout + global features
        # Output is 1: log(S/N)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_channels + hidden_channels // 2, hidden_channels // 2),
            LayerNorm(hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Select only the 4 node features you want
        node_features = torch.cat([
            x[:, self.feature_indices['position']],           # positions (x, y)
            x[:, self.feature_indices['rydberg_val']].unsqueeze(-1),  # Rydberg probability
            x[:, self.feature_indices['mask']].unsqueeze(-1),         # subsystem mask
        ], dim=1)

        # Select only the 2 edge features you want
        edge_features = torch.stack([
            edge_attr[:, 1],  # two-point correlation
            edge_attr[:, 2],  # distance
        ], dim=1)

        # Encode node & edge features
        x_enc = self.node_encoder(node_features)
        e_enc = self.edge_encoder(edge_features)

        # Message Passing with GATConv
        h = x_enc
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index, e_enc)
            h_new = self.norms[i](h_new)
            # Residual + dropout
            h = h + F.dropout(h_new, p=self.dropout_p, training=self.training)

        # Pool across the entire graph
        h_pool = global_add_pool(h, batch)
        # Pass it through the pool MLP
        h_readout = self.global_pool(h_pool)  # rename for clarity

        # Global features: nA/N, nB/N
        nA_over_N = data.nA.squeeze(-1) / (data.system_size.squeeze(-1) + 1e-10)
        nB_over_N = data.nB.squeeze(-1) / (data.system_size.squeeze(-1) + 1e-10)
        global_feats = torch.stack([nA_over_N, nB_over_N], dim=1)
        gf_out = self.global_mlp(global_feats)

        # Combine the readout + global features
        combined = torch.cat([h_readout, gf_out], dim=1)

        # Final output: log(S/N)
        out = self.final_mlp(combined).squeeze(-1)
        return out



# -----------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device, clip_grad=None):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Single-head output: log(S/N)
        pred_log_s_over_n = model(data)
        targets = data.y.squeeze()
        system_size = data.system_size.squeeze(-1)
        subsystem_size = data.nA.squeeze(-1)

        loss = criterion(pred_log_s_over_n, targets, system_size, subsystem_size)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        batch_size = data.num_graphs
        n_samples += batch_size
        total_loss += loss.item() * batch_size

    return total_loss / n_samples if n_samples > 0 else 0.0

@torch.no_grad()
def evaluate(model, loader, criterion, device, name='Eval'):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    all_preds_abs = []
    all_targets = []

    for data in loader:
        data = data.to(device)
        pred_log_s_over_n = model(data)

        targets = data.y.squeeze()
        system_size = data.system_size.squeeze(-1)
        subsystem_size = data.nA.squeeze(-1)

        # Compute loss
        loss = criterion(pred_log_s_over_n, targets, system_size, subsystem_size)
        batch_size = data.num_graphs
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        # Convert predicted log(S/N) -> absolute S
        pred_entropy_abs = torch.exp(pred_log_s_over_n)* system_size
        all_preds_abs.append(pred_entropy_abs.cpu())
        all_targets.append(targets.cpu())

    mean_loss = total_loss / n_samples if n_samples > 0 else 0.0

    # Metrics: MSE, MAE, MAPE for absolute S
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

    return {
        'loss': mean_loss,
        'mse_abs': mse_abs,
        'mae_abs': mae_abs,
        'mape_abs': mape_abs,
        'predictions_abs': all_preds_abs,
        'targets': all_targets
    }

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    setup_logging()
    set_seed(CONFIG['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Initialize the model
    model = ExperimentalGNN(
        hidden_channels=CONFIG['hidden_channels'],
        num_layers=10,
        dropout_p=CONFIG['dropout_p']
    ).to(device)

    criterion = PhysicalScaleAwareLoss(physics_weight=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['scheduler_factor'],
        patience=CONFIG['scheduler_patience']
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    best_val_loss = float('inf')

    for epoch in range(CONFIG['num_epochs']):
        logging.info(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_grad=CONFIG['grad_clip']
        )
        logging.info(f"  Training Loss: {train_loss:.6f}")

        val_metrics = evaluate(model, val_loader, criterion, device, name='Validation')
        val_loss = val_metrics['loss']
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['best_model_path'])
            logging.info(f"  [Info] Best model saved (val_loss={best_val_loss:.6f})")

    logging.info("Training complete. Loading best model for final validation...")
    model.load_state_dict(torch.load(CONFIG['best_model_path'], map_location=device))
    _ = evaluate(model, val_loader, criterion, device, name='Final Validation')


if __name__ == "__main__":
    main()