import os
import logging
import random
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from torch_geometric.data import InMemoryDataset, Data
from itertools import combinations

# Configuration
CONFIG = {
    'data_paths': [
        'Rydberg50k8-9.parquet',
        'Rydberg1.5M1-8.parquet',
    ],
    'processed_dir': './processed_experimentalc',
    'processed_file_name': 'data.pt',
    'random_seed': 42,
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

class ExperimentalSpinSystemDataset(InMemoryDataset):
    def __init__(self, dataframe, root='.', transform=None, pre_transform=None):
        self.df = dataframe
        super().__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            logging.info("Loading existing processed dataset...")
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            logging.info("Processing dataset from scratch...")
            self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [CONFIG['processed_file_name']]

    def download(self):
        pass

    def calculate_quantum_correlations(self, state_indices, state_probs, N):
        """Calculate full quantum 2-point correlation matrix."""
        correlations = torch.zeros((N, N), dtype=torch.float)
        
        # Calculate <ni> and <ninj>
        for state, prob in zip(state_indices, state_probs):
            state = int(state)
            # Single-site expectations
            for i in range(N):
                if (state & (1 << i)) != 0:
                    correlations[i, i] += prob
            
            # Two-site correlations
            for i, j in combinations(range(N), 2):
                if ((state & (1 << i)) != 0) and ((state & (1 << j)) != 0):
                    correlations[i, j] += prob
                    correlations[j, i] += prob
        
        # Calculate connected correlations: <ninj> - <ni><nj>
        for i in range(N):
            for j in range(i+1, N):
                connected_corr = correlations[i, j] - correlations[i, i] * correlations[j, j]
                correlations[i, j] = connected_corr
                correlations[j, i] = connected_corr
            
            # Set diagonal to single-site expectations
            correlations[i, i] = correlations[i, i]
            
        return correlations

    def process(self):
        data_list = []
        total_count = 0

        for idx, row in self.df.iterrows():
            total_count += 1
            
            # Basic system parameters
            Nx = row['Nx']
            Ny = 2
            N = Nx * Ny

            # Build Node Features - using unit spacing grid
            positions = np.array([
                (col, row_idx)  # Unit spacing grid
                for row_idx in range(Nx) for col in range(Ny)
            ], dtype=np.float32)
            positions_t = torch.tensor(positions, dtype=torch.float)

            # Calculate quantum correlations
            top_indices = row['Top_Indices']
            top_probs = row['Top_Probabilities']
            correlation_matrix = self.calculate_quantum_correlations(top_indices, top_probs, N)
            
            # Subsystem mask
            subsystem_mask_str = row['Subsystem_Mask']
            mask_tensor = torch.tensor([int(bit) for bit in subsystem_mask_str],
                                     dtype=torch.float).unsqueeze(1)

            # Combine node features
            node_features = torch.cat([
                positions_t,                            # 2 (geometric structure)
                correlation_matrix.diagonal().unsqueeze(1),  # 1 (quantum expectation)
                mask_tensor,                           # 1 (partition info)
            ], dim=1)

            # Build Edges - fully connected graph
            edges = list(combinations(range(N), 2))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Calculate edge attributes with normalized distances
            pos_i = positions_t[edge_index[0]]
            pos_j = positions_t[edge_index[1]]
            vec_ij = pos_j - pos_i
            # Normalize by sqrt(N) which is proportional to the maximum possible distance in a N-site system
            dist_ij = torch.norm(vec_ij, dim=1, keepdim=True) / np.sqrt(N)
            angle_ij = torch.atan2(vec_ij[:,1], vec_ij[:,0]).unsqueeze(1)
            
            # Use quantum correlations for edge attributes
            correlations = torch.tensor([correlation_matrix[i, j] for i, j in edges],
                                      dtype=torch.float).unsqueeze(1)

            edge_attr = torch.cat([
                angle_ij,     # [E,1] geometric
                correlations, # [E,1] quantum correlations
                dist_ij      # [E,1] distances
            ], dim=1)

            # Create Data object with Von Neumann Entropy as target
            target_vne = torch.tensor([row['Von_Neumann_Entropy']], dtype=torch.float)
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=target_vne,
            )

            # Add remaining properties
            data.system_size = torch.tensor([[N]], dtype=torch.float)
            
            nA_val = float(mask_tensor.sum().item())
            nB_val = N - nA_val
            data.nA = torch.tensor([[nA_val]], dtype=torch.float)
            data.nB = torch.tensor([[nB_val]], dtype=torch.float)

            data_list.append(data)

            if (idx+1) % 20000 == 0:
                logging.info(f"Processed {idx+1} rows so far...")

        # Final statistics
        logging.info(f"\nFinal statistics:")
        logging.info(f"Total samples processed: {total_count}")
        logging.info(f"Samples kept: {len(data_list)}")

        # Save dataset
        data_obj, slices = self.collate(data_list)
        torch.save((data_obj, slices), self.processed_paths[0])
        self.data, self.slices = data_obj, slices

def load_data():
    df_list = []
    for path in CONFIG['data_paths']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")
        df_temp = pq.read_table(path).to_pandas()
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    df_shuffled = df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    return ExperimentalSpinSystemDataset(dataframe=df_shuffled, root=CONFIG['processed_dir'])

def main():
    setup_logging()
    set_seed(CONFIG['random_seed'])
    dataset = load_data()
    logging.info(f"Finished processing. Dataset length: {len(dataset)}")
    logging.info(f"Sample data object: {dataset[0]}")

if __name__ == "__main__":
    main()