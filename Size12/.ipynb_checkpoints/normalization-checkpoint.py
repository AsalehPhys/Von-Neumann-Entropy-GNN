import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import (
    GINEConv,
    TransformerConv,
    Set2Set,
    LayerNorm
)

# Configuration
CONFIG = {
    'processed_dir': './processed_experimental12/processed',
    'processed_file_name': 'data.pt',
    'batch_size': 512,
    'hidden_channels': 512,
    'num_layers': 8,
    'dropout_p': 0.4
}

# Dataset Class (unchanged)
class SpinSystemDataset(InMemoryDataset):
    def __init__(self, root='.', transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        processed_path = os.path.join(root, CONFIG['processed_file_name'])
        self.data, self.slices = torch.load(processed_path, map_location=torch.device('cpu'), weights_only=False)
    
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

# Modified GNN with saved statistics
class ExperimentalGNNWithSavedStats(nn.Module):
    def __init__(self, hidden_channels=512, num_layers=8, dropout_p=0.4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        
        self.feature_indices = {
            'position': slice(0, 2),
            'rydberg_val': 2,
            'mask': 3,
        }

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

        self.convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i % 2 == 0:
                mp_mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    LayerNorm(hidden_channels),
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
                LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            ))
            self.norms.append(LayerNorm(hidden_channels))

        self.pool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            ) for _ in range(num_layers // 2)
        ])

        self.readout = nn.ModuleList([
            Set2Set(hidden_channels, processing_steps=4) for _ in range(2)
        ])
        
        self.readout_projection = nn.Sequential(
            nn.Linear(4 * hidden_channels, 2 * hidden_channels),
            LayerNorm(2 * hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p)
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels)
        )

        combined_dim = (2 * hidden_channels) + hidden_channels
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

        # Initialize storage for normalization statistics
        self.stored_stats = {}

    def save_normalization_stats(self, dataset, batch_size=512):
        """Compute and save running statistics from training data"""
        print("Computing normalization statistics...")
        self.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        device = next(self.parameters()).device

        # Initialize running statistics
        running_means = {}
        running_vars = {}
        count = 0

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

                # Node features
                node_features = torch.cat([
                    x[:, self.feature_indices['position']],
                    x[:, self.feature_indices['rydberg_val']].unsqueeze(-1),
                    x[:, self.feature_indices['mask']].unsqueeze(-1),
                ], dim=1)

                # Track statistics for each normalization layer
                h = self.node_encoder[0](node_features)
                if count == 0:
                    running_means['node_encoder'] = h.mean(0)
                    running_vars['node_encoder'] = h.var(0, unbiased=True)
                else:
                    running_means['node_encoder'] = (running_means['node_encoder'] * count + h.mean(0)) / (count + 1)
                    running_vars['node_encoder'] = (running_vars['node_encoder'] * count + h.var(0, unbiased=True)) / (count + 1)

                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} batches...")

        self.stored_stats = {
            'means': running_means,
            'vars': running_vars
        }
        
        print("Saving normalization statistics...")
        torch.save(self.stored_stats, 'normalization_stats.pt')
        print("Statistics saved successfully!")

    def load_normalization_stats(self, stats_path='normalization_stats.pt'):
        """Load pre-computed normalization statistics"""
        print("Loading normalization statistics...")
        self.stored_stats = torch.load(stats_path)
        print("Statistics loaded successfully!")

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Node feature encoding
        node_features = torch.cat([
            x[:, self.feature_indices['position']],
            x[:, self.feature_indices['rydberg_val']].unsqueeze(-1),
            x[:, self.feature_indices['mask']].unsqueeze(-1),
        ], dim=1)

        # Use saved statistics during evaluation
        if not self.training and hasattr(self, 'stored_stats'):
            h = self.node_encoder[0](node_features)
            h = (h - self.stored_stats['means']['node_encoder'].to(h.device)) / \
                (torch.sqrt(self.stored_stats['vars']['node_encoder'].to(h.device)) + 1e-5)
            h = self.node_encoder[2](h)
        else:
            h = self.node_encoder(node_features)

        edge_features = torch.stack([
            edge_attr[:, 0],
            edge_attr[:, 1],
            edge_attr[:, 2]
        ], dim=1)
        e_enc = self.edge_encoder(edge_features)

        for i in range(self.num_layers):
            e_enc = self.edge_convs[i](e_enc)
            h_new = self.convs[i](h, edge_index, e_enc)
            h_new = self.norms[i](h_new)
            h = h + h_new
            if i % 2 == 0 and i // 2 < len(self.pool):
                h = self.pool[i // 2](h)

        readouts = [readout(h, batch) for readout in self.readout]
        h_readout = torch.cat(readouts, dim=1)
        h_readout = self.readout_projection(h_readout)

        nA_over_N = data.nA.squeeze(-1) / (data.system_size.squeeze(-1) + 1e-10)
        nB_over_N = data.nB.squeeze(-1) / (data.system_size.squeeze(-1) + 1e-10)
        global_feats = torch.stack([nA_over_N, nB_over_N], dim=1)
        gf_out = self.global_mlp(global_feats)

        combined = torch.cat([h_readout, gf_out], dim=1)
        out = self.final_mlp(combined).squeeze(-1)
        return out

def evaluate_model(model, dataset, batch_size=1):
    """Evaluate model with specified batch size"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            target = data.y
            
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    return all_preds, all_targets, mse, mae

def main():
    # 1. Load your original training dataset
    print("Loading training dataset...")
    training_dataset = SpinSystemDataset(root=CONFIG['processed_dir'])
    
    # 2. Initialize the model with saved statistics capability
    print("Initializing model...")
    model = ExperimentalGNNWithSavedStats(
        hidden_channels=CONFIG['hidden_channels'],
        num_layers=CONFIG['num_layers'],
        dropout_p=CONFIG['dropout_p']
    )
    
    # 3. Load the trained model weights with CPU mapping
    print("Loading model weights...")
    model.load_state_dict(torch.load('best_model13.pth', map_location=torch.device('cpu')))
    
    # 4. Compute and save normalization statistics
    print("Computing normalization statistics...")
    model.save_normalization_stats(training_dataset)
    
    # 5. Load your evaluation dataset(s)
    print("Loading evaluation dataset...")
    eval_dataset = SpinSystemDataset(root=CONFIG['processed_dir'])  # Replace with your Rb/a=1.5 dataset
    
    # 6. Evaluate with batch_size=1
    print("Evaluating model...")
    predictions, targets, mse, mae = evaluate_model(model, eval_dataset, batch_size=1)
    
    print(f"Results with saved statistics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # 7. Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.savefig('prediction_results.png')
    plt.close()

if __name__ == "__main__":
    main()