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
    TransformerConv,  # Use TransformerConv instead of NNConv
    BatchNorm,
    GlobalAttention
)
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {
    'processed_dir': './processed/processed',
    'processed_file': 'processed/processed/data.pt',
    'scalers_path': 'scalers.pkl',
    'batch_size': 1024,
    'learning_rate': 5e-5,
    'weight_decay': 5e-3,
    'hidden_channels': 512,
    'num_epochs': 200,
    'patience': 20,
    'random_seed': 42,
    'best_model_path': 'best_gnn_model.pth',
    'loss_alpha': 1.0,
    'dropout_p': 0.3,
    'heads': 6,
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

class ImprovedGNNModel(nn.Module):
    def __init__(self, num_node_features, edge_attr_dim, hidden_channels, dropout_p=0.3, heads=6):
        super(ImprovedGNNModel, self).__init__()
        torch.manual_seed(CONFIG['random_seed'])

        # Using TransformerConv layers with multiple heads
        self.conv1 = TransformerConv(
            in_channels=num_node_features,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_attr_dim,
            dropout=dropout_p
        )
        self.bn1 = BatchNorm(hidden_channels * heads)

        self.conv2 = TransformerConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_attr_dim,
            dropout=dropout_p
        )
        self.bn2 = BatchNorm(hidden_channels * heads)

        self.conv3 = TransformerConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_attr_dim,
            dropout=dropout_p
        )
        self.bn3 = BatchNorm(hidden_channels * heads)

        self.dropout = nn.Dropout(p=dropout_p)

        gate_nn = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        self.global_pool = GlobalAttention(gate_nn=gate_nn)

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels * 2),
            BatchNorm(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_channels * 2, hidden_channels),
            BatchNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x2 = self.conv2(x, edge_index, edge_attr)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x = x + x2  # Residual

        x3 = self.conv3(x, edge_index, edge_attr)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x = x + x3  # Residual

        x = self.dropout(x)

        x_pooled = self.global_pool(x, batch)
        out = self.fc(x_pooled)
        return out.squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}')

def initialize_model(dataset, config):
    num_node_features = dataset[0].num_node_features
    edge_attr_dim = dataset[0].edge_attr.shape[1]
    hidden_channels = config['hidden_channels']
    model = ImprovedGNNModel(
        num_node_features=num_node_features,
        edge_attr_dim=edge_attr_dim,
        hidden_channels=hidden_channels,
        dropout_p=config['dropout_p'],
        heads=config['heads']
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

def train_epoch(model, optimizer, criterion, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        # out and data.y are log(entropy_per_Nx)
        loss = criterion(out, data.y.squeeze())
        loss.backward()
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
            log_pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(log_pred, data.y.squeeze())
            total_loss += loss.item() * data.num_graphs

            Nx = data.Nx
            y_true = torch.exp(data.y.squeeze()) * Nx
            y_pred = torch.exp(log_pred) * Nx

            mae_val = F.l1_loss(y_pred, y_true, reduction='sum').item()
            total_mae += mae_val

            nonzero_mask = (y_true != 0)
            if nonzero_mask.sum() > 0:
                percentage_errors = torch.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]) * 100
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
            log_out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            Nx = data.Nx
            pred_entropy = torch.exp(log_out) * Nx
            true_entropy = torch.exp(data.y.squeeze()) * Nx

            preds.append(pred_entropy.cpu().numpy())
            ys.append(true_entropy.cpu().numpy())

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
            f'Train Loss (log): {train_loss:.6f} | '
            f'Val Loss (log): {val_loss:.6f} | '
            f'Val MAE (original): {val_mae:.6f} | '
            f'Val MAPE (original): {val_percentage_error:.6f} | '
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
    test_loss, test_mae, test_pe = evaluate(model, criterion, test_loader)
    logging.info(f'\nTest Loss (log-scale): {test_loss:.6f}')
    logging.info(f'Test MAE (original scale): {test_mae:.6f}')
    logging.info(f'Test MAPE (original scale): {test_pe:.6f}')

    test_preds, test_ys = get_predictions(model, test_loader)
    mse = mean_squared_error(test_ys, test_preds)
    mae = mean_absolute_error(test_ys, test_preds)
    r2 = r2_score(test_ys, test_preds)

    logging.info(f'\nTest MSE (original): {mse:.6f}')
    logging.info(f'Test MAE (original): {mae:.6f}')
    logging.info(f'Test R2 (original): {r2:.6f}')

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=test_ys, y=test_preds, alpha=0.5)
    plt.plot([test_ys.min(), test_ys.max()], [test_ys.min(), test_ys.max()], 'r--')
    plt.xlabel('True von Neumann Entropy')
    plt.ylabel('Predicted von Neumann Entropy')
    plt.title('True vs. Predicted von Neumann Entropy')
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
