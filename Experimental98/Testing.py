
import os
import sys
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from torch_geometric.nn import (
    GINEConv,
    GlobalAttention,
    Set2Set,
    TransformerConv
)
from torch_geometric.nn.norm import BatchNorm

try:
    from diagnostic_plots import create_diagnostic_plots
except ImportError:
    def create_diagnostic_plots(*args, **kwargs):
        logging.info("diagnostic_plots.py not found; skipping plots.")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
CONFIG = {
    'processed_dir': './processed_experimental_8-9',
    'processed_file_name': 'data.pt',
    'batch_size': 1024,
    'best_model_path': 'best_experimental_gnn_98.pth',
    'hidden_channels': 512,
    'num_layers': 10,
    'dropout_p': 0.4
}

# -------------------------------------------------------------------
# Logging & Utilities
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class SpinSystemDataset(InMemoryDataset):
    """Loads the experimental dataset."""
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

# -------------------------------------------------------------------
# Experimental GNN Model
# -------------------------------------------------------------------
class ExperimentalGNN(nn.Module):
    """GNN using only experimentally accessible quantities."""
    def __init__(
        self,
        num_node_features,
        edge_attr_dim,
        hidden_channels=512,
        num_layers=10,
        dropout_p=0.4
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        # Node embedding
        self.init_transform = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
        )

        # Message Passing
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

        # Global transform (experimental features)
        self.global_transform = nn.Sequential(
            nn.Linear(7, hidden_channels),
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

        # Final MLP
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
            h = h + h_new

        # Graph-level readouts
        s2s = self.set2set_readout(h, batch)
        ga = self.global_attention(h, batch)

        # Experimental global features
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

# -------------------------------------------------------------------
# Evaluation Function
# -------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, name='Eval'):
    """Evaluates model and returns metrics compatible with diagnostic_plots."""
    model.eval()

    all_preds_abs = []
    all_preds_over_n = []
    all_targets = []
    all_sizes = []
    all_rydberg = []
    all_density = []

    for data in loader:
        data = data.to(device)
        preds = model(data)

        # Get predictions
        log_s_over_n = preds[:, 0]
        s_over_n = preds[:, 1]
        abs_pred = torch.exp(log_s_over_n * data.system_size.squeeze(-1))

        # Store results
        all_preds_abs.append(abs_pred.cpu())
        all_preds_over_n.append(s_over_n.cpu())
        all_targets.append(data.y.squeeze().cpu())
        all_sizes.append(data.system_size.squeeze().cpu())
        all_rydberg.append(data.rydberg_density.cpu())
        all_density.append(data.total_rydberg.cpu())

    # Concatenate everything
    predictions = torch.cat(all_preds_abs).numpy()
    predictions_over_n = torch.cat(all_preds_over_n).numpy()
    targets = torch.cat(all_targets).numpy()
    sizes = torch.cat(all_sizes).numpy()
    rydberg = torch.cat(all_rydberg).numpy()
    density = torch.cat(all_density).numpy()

    # Compute size-specific metrics
    unique_sizes = np.unique(sizes)
    size_metrics = {}
    for sz in unique_sizes:
        mask = (sizes == sz)
        mse_ = mean_squared_error(targets[mask], predictions[mask])
        mae_ = mean_absolute_error(targets[mask], predictions[mask])
        mape_ = np.mean(np.abs(predictions[mask] - targets[mask]) / (targets[mask] + 1e-10)) * 100
        size_metrics[int(sz)] = dict(mse=mse_, mae=mae_, mape=mape_)

    logging.info(f"\n[{name}] Evaluation metrics:")
    for sz, met in sorted(size_metrics.items()):
        logging.info(
            f"  Size={sz:2d} => MSE={met['mse']:.4e}, "
            f"MAE={met['mae']:.4e}, MAPE={met['mape']:.2f}%"
        )

    return {
        'predictions': predictions,
        'predictions_over_n': predictions_over_n,
        'targets': targets,
        'sizes': sizes,
        'rydberg_density': rydberg,
        'total_density': density,
        'size_metrics': size_metrics
    }

# -------------------------------------------------------------------
# Main Diagnostic Function
# -------------------------------------------------------------------
def run_diagnostics(N=5, save_plots=True, save_dir="plots"):
    """Runs model evaluation and creates diagnostic plots."""
    setup_logging()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = SpinSystemDataset(root=CONFIG['processed_dir'])
    if len(dataset) == 0:
        logging.error("Dataset is empty. Exiting.")
        return

    # Take N samples
    if N < len(dataset):
        dataset = dataset[:N]
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'])

    # Initialize model
    sample_data = next(iter(loader))
    model = ExperimentalGNN(
        num_node_features=sample_data.x.size(1),
        edge_attr_dim=sample_data.edge_attr.size(1),
        hidden_channels=CONFIG['hidden_channels'],
        num_layers=CONFIG['num_layers'],
        dropout_p=CONFIG['dropout_p']
    ).to(device)

    # Load weights
    if not os.path.exists(CONFIG['best_model_path']):
        logging.error(f"Model weights not found at '{CONFIG['best_model_path']}'")
        return
    logging.info(f"Loading model from '{CONFIG['best_model_path']}'")
    model.load_state_dict(torch.load(CONFIG['best_model_path'], map_location=device))

    # Run evaluation
    diagnostics = evaluate(model, loader, device)

    # Create plots
    if save_plots:
        create_diagnostic_plots(diagnostics, save_plots=True, save_dir=save_dir)
    
    logging.info("Diagnostics completed.")
    return diagnostics

if __name__ == "__main__":
    _ = run_diagnostics(N=50000, save_plots=True, save_dir="diagnostic_plots")