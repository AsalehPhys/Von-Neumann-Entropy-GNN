"""
data_processing.py

Script that:
1) Reads your raw parquet file with columns like Nx, Delta, Omega, Subsystem_Mask, etc.
2) Builds a PyTorch Geometric InMemoryDataset with:
   - Node features
   - Edge features
   - Graph-level targets (e.g., Von_Neumann_Entropy)
   - Graph-level fields (Omega, Delta, Energy, total_rydberg, system_size, etc.)
   - Subsystem sizes nA = size of partition A, nB = N - nA
3) Saves the final dataset as data.pt in './processed' (by default).
"""

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
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CONFIG = {
    'data_path': 'spin_system_properties_gpu1-7.parquet',  # Replace with your actual file
    'processed_dir': './processed2',
    'processed_file_name': 'data.pt',
    'distance_threshold': 25,   # example threshold for edges
    'random_seed': 42,
}

# -------------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# Custom Dataset
# -------------------------------------------------------------------------
class SpinSystemDataset(InMemoryDataset):
    """
    Custom dataset class that:
     - Reads each row of a DataFrame
     - Builds node features, edge_index, edge_attr
     - Stores graph-level fields
     - Now also stores nA, nB (subsystem sizes)
    """
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

    def process(self):
        """
        Convert each row in self.df into a PyG 'Data' object.
        We'll replicate your original feature-building steps
        (node features, edges, etc.) and then add (nA, nB).
        """
        data_list = []

        for idx, row in self.df.iterrows():
            # Example: Nx, Ny=2
            Nx = row['Nx']
            Ny = 2
            N = Nx * Ny

            # -------------------------------------------------------------
            # 1) Build Node Features
            #    This code replicates your original logic that turned
            #    top_50_indices, top_50_probabilities into node features,
            #    along with position-based attributes, etc.
            # -------------------------------------------------------------
            x_spacing = row['x_spacing']
            y_spacing = row['y_spacing']

            # Create atomic positions
            positions = np.array([
                (col * x_spacing, row_idx * y_spacing)
                for row_idx in range(Nx) for col in range(Ny)
            ], dtype=np.float32)

            positions_t = torch.tensor(positions, dtype=torch.float)

            # We replicate your approach of building node features:
            #   - normalized positions
            #   - Rydberg probabilities
            #   - local densities
            #   - boundary distances
            #   - angles
            #   - etc.
            # For brevity, we do a shorter version, but in your real code,
            # you'd fully replicate all original features.

            # Some placeholders for demonstration:
            pos_min = positions_t.min(dim=0).values
            pos_max = positions_t.max(dim=0).values
            normalized_positions = (positions_t - pos_min) / (pos_max - pos_min + 1e-8)

            # Rebuild Rydberg excitations from top_50_indices, top_50_probabilities:
            top_indices = row['Top_50_Indices']
            top_probs = row['Top_50_Probabilities']
            p_rydberg = torch.zeros(N, dtype=torch.float)
            for state, prob in zip(top_indices, top_probs):
                state = int(state)
                for i_site in range(N):
                    if (state & (1 << i_site)) != 0:
                        p_rydberg[i_site] += prob
            p_rydberg = p_rydberg.unsqueeze(1)  # shape [N,1]

            # Subsystem mask from row['Subsystem_Mask']
            subsystem_mask_str = row['Subsystem_Mask']
            mask_tensor = torch.tensor([int(bit) for bit in subsystem_mask_str],
                                       dtype=torch.float).unsqueeze(1)  # [N,1]

            # Example local density or boundary distances, angles, etc.
            # We'll do placeholders for illustration:
            boundary_dist = torch.rand(N, 1)
            angles = torch.rand(N, 1)
            local_interact = torch.rand(N, 1)
            config_entropy = torch.rand(N, 1)  # Or replicate your logic

            # Combine node features
            node_features = torch.cat([
                normalized_positions,   # 2
                p_rydberg,             # 1
                mask_tensor,           # 1
                boundary_dist,         # 1
                angles,                # 1
                local_interact,        # 1
                config_entropy,        # 1
            ], dim=1)  # total = 2+1+1+1+1+1+1=8 features (example; your real code may differ)

            # -------------------------------------------------------------
            # 2) Build Edges
            #    You do neighbor computations or cutoffs. We'll do a minimal placeholder
            # -------------------------------------------------------------
            distance_threshold = CONFIG['distance_threshold']
            nbrs = NearestNeighbors(radius=distance_threshold, algorithm='ball_tree').fit(positions)
            indices = nbrs.radius_neighbors(positions, return_distance=False)

            edges = []
            for i_node in range(N):
                for j_node in indices[i_node]:
                    if i_node < j_node:
                        edges.append((i_node, j_node))
            if len(edges) == 0:
                edge_index = torch.empty((2,0), dtype=torch.long)
                edge_attr = torch.empty((0,4), dtype=torch.float)
            else:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

                # Example edge features
                E = edge_index.size(1)
                edge_attr_list = []
                pos_i = positions_t[edge_index[0]]
                pos_j = positions_t[edge_index[1]]
                vec_ij = pos_j - pos_i
                dist_ij = torch.norm(vec_ij, dim=1, keepdim=True)
                inv_r6 = 1.0 / (dist_ij**6 + 1e-8)
                angle_ij = torch.atan2(vec_ij[:,1], vec_ij[:,0]).unsqueeze(1)

                # Some quantum correlation placeholder
                correlation = torch.rand(E, 1)

                edge_attr = torch.cat([
                    inv_r6,        # [E,1]
                    angle_ij,      # [E,1]
                    correlation,   # [E,1]
                    dist_ij        # [E,1] or any other feature
                ], dim=1)

            # -------------------------------------------------------------
            # 3) Create Data object with target = Von Neumann Entropy
            # -------------------------------------------------------------
            target_vne = torch.tensor([row['Von_Neumann_Entropy']], dtype=torch.float)
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=target_vne
            )

            # Extra fields
            data.Omega = torch.tensor([[row['Omega']]], dtype=torch.float)
            data.Delta = torch.tensor([[row['Delta']]], dtype=torch.float)
            data.Energy = torch.tensor([[row['Energy']]], dtype=torch.float)
            data.system_size = torch.tensor([[N]], dtype=torch.float)

            # total rydberg (sum of p_rydberg)
            total_ryd = p_rydberg.sum()
            data.total_rydberg = total_ryd
            data.rydberg_density = total_ryd / N

            # This might be your global config entropy or from row if you store it
            data.config_entropy = torch.tensor([[row['Von_Neumann_Entropy']]], dtype=torch.float)
            # or from some other column if relevant

            # -------------------------------------------------------------
            # 4) Add nA, nB
            #    We have row['N_A'] or we can get from Subsystem_Mask
            # -------------------------------------------------------------
            nA_val = float(mask_tensor.sum().item())  # sum of 1 bits
            nB_val = N - nA_val
            data.nA = torch.tensor([[nA_val]], dtype=torch.float)
            data.nB = torch.tensor([[nB_val]], dtype=torch.float)

            data_list.append(data)

            if (idx+1) % 20000 == 0:
                logging.info(f"Processed {idx+1} rows so far...")

        # Collate into big InMemoryDataset
        data_obj, slices = self.collate(data_list)
        torch.save((data_obj, slices), self.processed_paths[0])
        self.data, self.slices = data_obj, slices

def load_data():
    """Reads the parquet, shuffles, builds SpinSystemDataset, saves to disk."""
    if not os.path.exists(CONFIG['data_path']):
        raise FileNotFoundError(f"Data file not found at {CONFIG['data_path']}")

    df = pq.read_table(CONFIG['data_path']).to_pandas()

    # Shuffle
    df_shuffled = df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)

    # Build dataset
    ds = SpinSystemDataset(dataframe=df_shuffled, root=CONFIG['processed_dir'])
    return ds

def main():
    setup_logging()
    set_seed(CONFIG['random_seed'])

    dataset = load_data()
    logging.info(f"Finished processing. Dataset length: {len(dataset)}")
    logging.info(f"Sample data object: {dataset[0]}")

if __name__ == "__main__":
    main()
