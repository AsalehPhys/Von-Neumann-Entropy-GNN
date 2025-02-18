import os
import sys
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import optuna
from optuna.exceptions import TrialPruned

# PyTorch Geometric imports (using the PyG DataLoader!)
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader  # PyG DataLoader supports batching Data objects
from torch_geometric.nn import (
    GINEConv,
    TransformerConv,
    Set2Set,
    BatchNorm
)
from torch_geometric.utils import dropout_adj

from sklearn.metrics import mean_squared_error, mean_absolute_error

###############################################################################
# Global Configuration
###############################################################################
CONFIG = {
    'processed_dir': './processed_experimentalrung1_6',
    'processed_file_name': 'data.pt',
    'batch_size': 256,
    'num_epochs': 100,   # Adjust epochs as needed
    'patience': 20,      # Early stopping patience
    'random_seed': 42,
    'grad_clip': 1.0
}

###############################################################################
# Logging & Reproducibility
###############################################################################
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

###############################################################################
# SpinSystemDataset (PyG InMemoryDataset)
###############################################################################
class SpinSystemDataset(InMemoryDataset):
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

###############################################################################
# PhysicalScaleAwareLoss
###############################################################################
class PhysicalScaleAwareLoss(nn.Module):
    def __init__(self, physics_weight=0.5, rel_weight=0.0, eps=1e-10):
        super().__init__()
        self.physics_weight = physics_weight
        self.rel_weight = rel_weight
        self.eps = eps

    def get_entropy_bounds(self, system_size, subsystem_size):
        lower_bound = torch.zeros_like(system_size, dtype=torch.float)
        min_size = torch.minimum(subsystem_size.float(),
                                 (system_size - subsystem_size).float())
        upper_bound = min_size * torch.log(torch.tensor(2.0, device=system_size.device))
        return lower_bound, upper_bound

    def forward(self, pred_s, target_s, system_size, subsystem_size):
        error = pred_s - target_s
        logcosh = torch.log(torch.cosh(error + self.eps)).mean()

        rel_error = torch.abs(error) / (torch.abs(target_s) + self.eps)
        rel_loss = rel_error.mean()

        lower_bound, upper_bound = self.get_entropy_bounds(system_size, subsystem_size)
        bounds_violation = (
            torch.relu(lower_bound - pred_s) + torch.relu(pred_s - upper_bound)
        ).mean()

        total_loss = (1 - self.rel_weight) * logcosh + \
                     self.rel_weight * rel_loss + \
                     self.physics_weight * bounds_violation
        return total_loss

###############################################################################
# ExperimentalGNN Model
###############################################################################
class ExperimentalGNN(nn.Module):
    def __init__(self, hidden_channels=64, num_layers=6, dropout_p=0.4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.feature_indices = {
            'position': slice(0, 2),  # first two columns
            'rydberg_val': 2,         # third column
            'mask': 3,                # fourth column
        }

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(4, hidden_channels),
            BatchNorm(hidden_channels),
            nn.SiLU()
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_channels),
            BatchNorm(hidden_channels),
            nn.SiLU()
        )

        # Edge attention modules
        self.edge_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])

        self.convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i % 2 == 0:
                mp_mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    BatchNorm(hidden_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                conv = GINEConv(mp_mlp, edge_dim=hidden_channels)
            else:
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
                BatchNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            ))
            self.norms.append(BatchNorm(hidden_channels))

        # Edge-preserving pooling
        self.pool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            ) for _ in range(num_layers // 2)
        ])

        # Multi-head readout
        self.readout = nn.ModuleList([
            Set2Set(hidden_channels, processing_steps=4) for _ in range(2)
        ])

        self.readout_projection = nn.Sequential(
            nn.Linear(4 * hidden_channels, 2 * hidden_channels),
            BatchNorm(2 * hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p)
        )

        # Global features MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            BatchNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels)
        )

        combined_dim = (2 * hidden_channels) + hidden_channels
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels),
            BatchNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels // 2),
            BatchNorm(hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Node & edge feature encoding
        node_features = torch.cat([
            x[:, self.feature_indices['position']],
            x[:, self.feature_indices['rydberg_val']].unsqueeze(-1),
            x[:, self.feature_indices['mask']].unsqueeze(-1),
        ], dim=1)

        x_enc = self.node_encoder(node_features)
        e_enc = self.edge_encoder(edge_attr)

        h = x_enc
        for i in range(self.num_layers):
            edge_weights = self.edge_attention[i](e_enc).squeeze(-1)
            e_enc = self.edge_convs[i](e_enc)
            h_new = self.convs[i](h, edge_index, e_enc * edge_weights.unsqueeze(-1))
            h_new = self.norms[i](h_new)
            h = h + h_new  # residual connection
            if i % 2 == 0 and (i // 2) < len(self.pool):
                h = self.pool[i // 2](h)

        readouts = [rd(h, batch) for rd in self.readout]
        h_readout = torch.cat(readouts, dim=1)
        h_readout = self.readout_projection(h_readout)

        # Global features
        nA_over_N = data.nA.squeeze(-1) / (data.system_size.squeeze(-1) + 1e-10)
        nB_over_N = data.nB.squeeze(-1) / (data.system_size.squeeze(-1) + 1e-10)
        global_feats = torch.stack([nA_over_N, nB_over_N], dim=1)
        gf_out = self.global_mlp(global_feats)

        combined = torch.cat([h_readout, gf_out], dim=1)
        out = self.final_mlp(combined).squeeze(-1)
        return out

###############################################################################
# Weight Initialization
###############################################################################
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

###############################################################################
# Training & Evaluation Functions
###############################################################################
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        pred_s = model(data)
        targets = data.y.squeeze()
        system_size = data.system_size.squeeze(-1)
        subsystem_size = data.nA.squeeze(-1)

        loss = criterion(pred_s, targets, system_size, subsystem_size)
        loss.backward()

        if CONFIG['grad_clip'] is not None:
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])

        optimizer.step()
        bs = data.num_graphs
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / (total_samples + 1e-10)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    for data in loader:
        data = data.to(device)
        pred_s = model(data)
        targets = data.y.squeeze()
        system_size = data.system_size.squeeze(-1)
        subsystem_size = data.nA.squeeze(-1)

        loss = criterion(pred_s, targets, system_size, subsystem_size)
        bs = data.num_graphs
        total_loss += loss.item() * bs
        total_samples += bs

        all_preds.append(pred_s.cpu())
        all_targets.append(targets.cpu())

    mean_loss = total_loss / (total_samples + 1e-10)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    mse_abs = mean_squared_error(all_targets, all_preds)
    mae_abs = mean_absolute_error(all_targets, all_preds)

    return mean_loss, mse_abs, mae_abs

###############################################################################
# Data Splitting Helper
###############################################################################
def get_data_splits():
    dataset = SpinSystemDataset(root=CONFIG['processed_dir'])
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )
    return train_ds, val_ds

###############################################################################
# DDP Worker Function (runs on each GPU / rank)
###############################################################################
def ddp_worker(rank, world_size, hyperparams, return_dict):
    setup_logging()
    set_seed(CONFIG['random_seed'])

    # Initialize process group (for a single-node multi-GPU run)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Create datasets and DistributedSamplers
    train_ds, val_ds = get_data_splits()
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    # Use the PyG DataLoader to batch graph Data objects
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], sampler=val_sampler)

    # Build model with hyperparameters
    model = ExperimentalGNN(
        hidden_channels=hyperparams['hidden_channels'],
        num_layers=hyperparams['num_layers'],
        dropout_p=hyperparams['dropout_p']
    )
    model.apply(init_weights)
    model.to(device)

    ddp_model = DDP(model, device_ids=[rank])

    criterion = PhysicalScaleAwareLoss(physics_weight=0.5, rel_weight=0.0)
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=hyperparams['T_0'],
        T_mult=hyperparams['T_mult'],
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(CONFIG['num_epochs']):
        train_sampler.set_epoch(epoch)

        train_loss = train_epoch(ddp_model, train_loader, optimizer, criterion, device)
        val_loss, val_mse, val_mae = evaluate(ddp_model, val_loader, criterion, device)

        scheduler.step()

        if rank == 0:
            logging.info(f"Epoch {epoch+1}/{CONFIG['num_epochs']} - "
                         f"Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, "
                         f"MSE: {val_mse:.6f}, MAE: {val_mae:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG['patience']:
                break

    if rank == 0:
        return_dict['best_val_loss'] = best_val_loss

    dist.barrier()
    dist.destroy_process_group()

###############################################################################
# Optuna Objective Function
###############################################################################
def objective(trial):
    # Sample hyperparameters with Optuna
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [256, 512, 1024])
    dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5, step=0.1)
    num_layers = trial.suggest_categorical("num_layers", [4, 6, 8])
    T_0 = trial.suggest_int("T_0", 10, 50, step=10)
    T_mult = trial.suggest_int("T_mult", 1, 3)

    hyperparams = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'hidden_channels': hidden_channels,
        'dropout_p': dropout_p,
        'num_layers': num_layers,
        'T_0': T_0,
        'T_mult': T_mult
    }

    # Log the parameters at the start of this trial.
    logging.info(f"Starting Trial {trial.number} with hyperparameters: {hyperparams}")

    # Use a Manager dict to share results from DDP worker (rank 0)
    manager = mp.Manager()
    return_dict = manager.dict()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found. This code requires at least one GPU.")

    mp.spawn(
        ddp_worker,
        nprocs=world_size,
        args=(world_size, hyperparams, return_dict)
    )

    best_val_loss = return_dict.get('best_val_loss', float('inf'))
    if best_val_loss == float('inf'):
        raise TrialPruned("No valid val_loss was returned from the DDP worker.")

    # Log the best validation loss for this trial.
    logging.info(f"Trial {trial.number} finished with best validation loss: {best_val_loss:.6f}")

    trial.set_user_attr("best_val_loss", best_val_loss)
    return best_val_loss

###############################################################################
# Main Entry Point
###############################################################################
def main():
    setup_logging()
    set_seed(CONFIG['random_seed'])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Adjust the number of trials as needed

    logging.info(f"Study complete. Best trial val_loss: {study.best_value:.6f}")
    logging.info(f"Best hyperparameters: {study.best_params}")

if __name__ == "__main__":
    main()
