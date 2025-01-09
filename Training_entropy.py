import os
import random
import time
import logging
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import (
    GINEConv,
    Set2Set,
    GraphNorm
)
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {
    'processed_dir': './processed/processed',
    'processed_file': 'processed/processed/data.pt',
    'batch_size': 2048,
    'learning_rate': 5e-5,
    'weight_decay': 1e-3,  
    'hidden_channels': 512,
    'num_epochs': 300,
    'patience': 30,
    'random_seed': 42,
    'best_model_path': 'best_gnn_model_improved.pth',
    'dropout_p': 0.6
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

setup_logging()
set_seed(CONFIG['random_seed'])

class SpinSystemDataset(InMemoryDataset):
    def __init__(self, root='.', processed_file='data.pt', transform=None, pre_transform=None):
        self.processed_file = processed_file
        super(SpinSystemDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(os.path.join(root, processed_file))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.processed_file]

    def download(self):
        pass

    def process(self):
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GraphNorm, Set2Set, GATConv  # Assuming GATConv for attention

class ImprovedGNNModel(nn.Module):
    def __init__(self, num_node_features, edge_attr_dim, hidden_channels, dropout_p=0.5, global_feature_dim=9):
        super(ImprovedGNNModel, self).__init__()

        # Enhanced MLP for GINEConv with additional layers
        def mlp(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, hidden_channels * 2),
                nn.ReLU(),
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels)
            )

        # GINEConv layers with edge_dim specified
        self.conv1 = GINEConv(mlp(num_node_features, hidden_channels), edge_dim=edge_attr_dim)
        self.norm1 = GraphNorm(hidden_channels)
        
        self.conv2 = GINEConv(mlp(hidden_channels, hidden_channels), edge_dim=edge_attr_dim)
        self.norm2 = GraphNorm(hidden_channels)
        
        self.conv3 = GINEConv(mlp(hidden_channels, hidden_channels), edge_dim=edge_attr_dim)
        self.norm3 = GraphNorm(hidden_channels)
        
        # Additional convolutional layers for increased depth
        self.conv4 = GINEConv(mlp(hidden_channels, hidden_channels), edge_dim=edge_attr_dim)
        self.norm4 = GraphNorm(hidden_channels)
        
        self.conv5 = GINEConv(mlp(hidden_channels, hidden_channels), edge_dim=edge_attr_dim)
        self.norm5 = GraphNorm(hidden_channels)

        self.dropout = nn.Dropout(p=dropout_p)

        # Set2Set for global readout with increased processing steps
        self.readout = Set2Set(hidden_channels, processing_steps=4, num_layers=3)

        # Incorporate global features by increasing input dimension of the first fc layer
        # Also, adding attention mechanism if desired
        input_dim = 2 * hidden_channels + global_feature_dim
        self.attention = nn.MultiheadAttention(embed_dim=2 * hidden_channels, num_heads=4, dropout=dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4 * hidden_channels),
            GraphNorm(4 * hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(4 * hidden_channels, 2 * hidden_channels),
            GraphNorm(2 * hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(2 * hidden_channels, hidden_channels),
            GraphNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch, global_features):
        # Layer 1
        h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(h, batch)
        h = F.relu(h)
        h1 = h

        # Layer 2
        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h, batch)
        h = F.relu(h)
        h = h + h1  # Residual
        h2 = h

        # Layer 3
        h = self.conv3(h, edge_index, edge_attr)
        h = self.norm3(h, batch)
        h = F.relu(h)
        h = h + h2  # Residual
        h3 = h

        # Layer 4
        h = self.conv4(h, edge_index, edge_attr)
        h = self.norm4(h, batch)
        h = F.relu(h)
        h = h + h3  # Residual
        h4 = h

        # Layer 5
        h = self.conv5(h, edge_index, edge_attr)
        h = self.norm5(h, batch)
        h = F.relu(h)
        h = h + h4  # Residual

        h = self.dropout(h)

        # Global readout
        h = self.readout(h, batch)  # [num_graphs, 2*hidden_channels]

        # Attention mechanism (optional)
        # If you want to apply attention between readout and global features
        # h = h.unsqueeze(0)  # Shape [1, num_graphs, 2*hidden_channels]
        # global_features = global_features.unsqueeze(0)  # Shape [1, num_graphs, global_feature_dim]
        # attn_output, _ = self.attention(h, h, h)
        # h = attn_output.squeeze(0)

        # Concatenate global features
        # global_features should be [num_graphs, global_feature_dim]
        h = torch.cat([h, global_features], dim=-1)

        # Fully connected MLP
        out = self.fc(h)
        return out.squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}')

def initialize_model(dataset, config):
    num_node_features = dataset[0].num_node_features
    edge_attr_dim = dataset[0].edge_attr.shape[1]
    hidden_channels = config['hidden_channels']

    # global_feature_dim is fixed at 9 (as defined in previous dataset code)
    global_feature_dim = dataset[0].global_features.shape[1]

    model = ImprovedGNNModel(
        num_node_features=num_node_features,
        edge_attr_dim=edge_attr_dim,
        hidden_channels=hidden_channels,
        dropout_p=config['dropout_p'],
        global_feature_dim=global_feature_dim
    ).to(device)
    logging.info("\nModel Architecture:")
    logging.info(model)
    return model

def setup_training(model, config):
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    return criterion, optimizer, scheduler

def train_epoch(model, optimizer, criterion, loader, clip=2.0):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_features)
        loss = criterion(out, data.y.squeeze())
        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_percentage_error = 0
    count = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_features)
            loss = criterion(pred, data.y.squeeze())
            total_loss += loss.item() * data.num_graphs

            y_true = data.y.squeeze()
            y_pred = pred

            total_mae += F.l1_loss(y_pred, y_true, reduction='sum').item()

            # MAPE
            nonzero_mask = (y_true != 0)
            if nonzero_mask.sum() > 0:
                percentage_errors = (torch.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
                total_percentage_error += percentage_errors.sum().item()
                count += nonzero_mask.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_mae = total_mae / len(loader.dataset)
    avg_percentage_error = total_percentage_error / count if count > 0 else float('nan')

    return avg_loss, avg_mae, avg_percentage_error

def get_predictions(model, loader):
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_features)
            preds.append(out.cpu().numpy())
            ys.append(data.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(ys)

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, config):
    best_val_loss = float('inf')
    patience = config['patience']
    trigger_times = 0

    logging.info("\nStarting Training...\n")

    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        val_loss, val_mae, val_percentage_error = evaluate(model, criterion, val_loader)
        scheduler.step()

        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        logging.info(
            f'Epoch [{epoch}/{config["num_epochs"]}] | '
            f'Train Loss: {train_loss:.6f} | '
            f'Val Loss: {val_loss:.6f} | '
            f'Val MAE: {val_mae:.6f} | '
            f'PE Error: {val_percentage_error:.6f} | '
            f'LR: {current_lr:.6f} | '
            f'Epoch Time: {epoch_duration:.2f} sec'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), config['best_model_path'])  
            logging.info(f'Best model saved at epoch {epoch}')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logging.info("Early stopping triggered!")
                break

    logging.info("\nTraining Completed.")

def test_model(model, criterion, test_loader, config):
    model.load_state_dict(torch.load(config['best_model_path']))
    test_loss, test_mae, test_perc_error = evaluate(model, criterion, test_loader)
    logging.info(f'\nTest Loss (Smooth L1 Loss): {test_loss:.6f}')
    logging.info(f'Test MAE: {test_mae:.6f}')
    logging.info(f'Test Pct Error: {test_perc_error:.6f}')

    test_preds, test_ys = get_predictions(model, test_loader)
    mse = mean_squared_error(test_ys, test_preds)
    mae = mean_absolute_error(test_ys, test_preds)
    r2 = r2_score(test_ys, test_preds)

    logging.info(f'\nTest MSE: {mse:.6f}')
    logging.info(f'Test MAE: {mae:.6f}')
    logging.info(f'Test R2: {r2:.6f}')

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=test_ys, y=test_preds, alpha=0.5)
    plt.plot([test_ys.min(), test_ys.max()], [test_ys.min(), test_ys.max()], 'r--')
    plt.xlabel('True Entropy')
    plt.ylabel('Predicted Entropy')
    plt.title('True vs. Predicted Entropy')
    plt.grid(True)
    plt.show()

def main():
    dataset = SpinSystemDataset(
        root=CONFIG['processed_dir'],
        processed_file='data.pt'
    )

    logging.info(f'\nTotal graphs in dataset: {len(dataset)}')
    logging.info(f'\nSample Data Object:')
    logging.info(dataset[0])

    total_length = len(dataset)
    train_length = int(0.8 * total_length)
    val_length = int(0.1 * total_length)
    test_length = total_length - train_length - val_length

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_length, val_length, test_length],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )

    logging.info(f'\nTraining graphs: {len(train_dataset)}')
    logging.info(f'Validation graphs: {len(val_dataset)}')
    logging.info(f'Test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    logging.info("DataLoaders created.")

    model = initialize_model(dataset, CONFIG)
    criterion, optimizer, scheduler = setup_training(model, CONFIG)
    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, CONFIG)
    test_model(model, criterion, test_loader, CONFIG)

if __name__ == "__main__":
    main()
