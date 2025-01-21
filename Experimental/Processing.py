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

# Configuration
CONFIG = {
    'data_paths': [
        'spin_system_properties_gpu1-7.parquet',
        'spin_system_properties_cpu_test8-9.parquet',
        'spin_system_properties_cpu_test50k.parquet',
        'spin_system_properties_cpu_test10k8-9.parquet',
        'filtered_spin_system_properties_cpu.parquet',
    ],
    'processed_dir': './processed_experimental',
    'processed_file_name': 'data.pt',
    'distance_threshold': 25,   # example threshold for edges
    'random_seed': 42,
    'prob_threshold': 0.9,     # minimum probability coverage required
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

def calculate_classical_mutual_information(p_rydberg, top_indices, top_probs, subsystem_mask, N):
    """Calculate mutual information using I(A:B) = ∑ₐ,ᵦ P(a,b) ln[P(a,b)/(P(a)P(b))]"""
    # Check if probabilities sum to ≥90%
    total_prob = sum(top_probs)
    if total_prob < CONFIG['prob_threshold']:
        return torch.tensor([float('nan')])
    
    # Split into subsystems
    subsys_A = set([i for i in range(N) if subsystem_mask[i] == 1])
    subsys_B = set([i for i in range(N) if subsystem_mask[i] == 0])
    
    # Initialize probability dictionaries
    P_joint = {}  # P(a,b)
    P_A = {}      # P(a)
    P_B = {}      # P(b)
    
    # Calculate joint and marginal probabilities
    for state_idx, prob in zip(top_indices, top_probs):
        state_bin = format(int(state_idx), f'0{N}b')
        config_A = ''.join(state_bin[i] for i in subsys_A)
        config_B = ''.join(state_bin[i] for i in subsys_B)
        
        key = (config_A, config_B)
        P_joint[key] = P_joint.get(key, 0) + prob
        P_A[config_A] = P_A.get(config_A, 0) + prob
        P_B[config_B] = P_B.get(config_B, 0) + prob
    
    # Calculate mutual information using natural logarithm
    I_AB = 0.0
    for (config_A, config_B), p_ab in P_joint.items():
        p_a = P_A[config_A]
        p_b = P_B[config_B]
        if p_ab > 0 and p_a > 0 and p_b > 0:
            I_AB += p_ab * torch.log(torch.tensor(p_ab / (p_a * p_b)))
    
    return torch.tensor([I_AB])

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

    def process(self):
        data_list = []
        skipped_count = 0
        total_count = 0
        prob_dist_stats = []

        for idx, row in self.df.iterrows():
            total_count += 1
            
            # Basic system parameters
            Nx = row['Nx']
            Ny = 2
            N = Nx * Ny

            # Build Node Features
            x_spacing = row['x_spacing']
            y_spacing = row['y_spacing']
            min_spacing = min(x_spacing, y_spacing)
            positions = np.array([
                (col * x_spacing/min_spacing, row_idx * y_spacing/min_spacing)
                for row_idx in range(Nx) for col in range(Ny)
            ], dtype=np.float32)
            positions_t = torch.tensor(positions, dtype=torch.float)

            # Rebuild Rydberg excitations
            top_indices = row['Top_50_Indices']
            top_probs = row['Top_50_Probabilities']
            prob_dist_stats.append(sum(top_probs))
            
            p_rydberg = torch.zeros(N, dtype=torch.float)
            for state, prob in zip(top_indices, top_probs):
                state = int(state)
                for i_site in range(N):
                    if (state & (1 << i_site)) != 0:
                        p_rydberg[i_site] += prob
            p_rydberg = p_rydberg.unsqueeze(1)

            # Subsystem mask
            subsystem_mask_str = row['Subsystem_Mask']
            mask_tensor = torch.tensor([int(bit) for bit in subsystem_mask_str],
                                     dtype=torch.float).unsqueeze(1)

            # Local features
            boundary_dist = torch.zeros(N, 1)
            for i in range(N):
                mask_i = int(subsystem_mask_str[i])
                min_dist = N
                for j in range(N):
                    if int(subsystem_mask_str[j]) != mask_i:
                        dist = abs(i - j)
                        min_dist = min(min_dist, dist)
                boundary_dist[i] = min_dist

            # Combine node features
            node_features = torch.cat([
                positions_t,      # 2 (geometric structure)
                p_rydberg,       # 1 (experimental)
                mask_tensor,     # 1 (partition info)
                boundary_dist,   # 1 (geometric)
            ], dim=1)

            # Build Edges
            nbrs = NearestNeighbors(radius=CONFIG['distance_threshold'], 
                                   algorithm='ball_tree').fit(positions)
            indices = nbrs.radius_neighbors(positions, return_distance=False)

            edges = []
            for i_node in range(N):
                for j_node in indices[i_node]:
                    if i_node < j_node:
                        edges.append((i_node, j_node))
                        
            if len(edges) == 0:
                edge_index = torch.empty((2,0), dtype=torch.long)
                edge_attr = torch.empty((0,3), dtype=torch.float)
            else:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                pos_i = positions_t[edge_index[0]]
                pos_j = positions_t[edge_index[1]]
                vec_ij = pos_j - pos_i
                dist_ij = torch.norm(vec_ij, dim=1, keepdim=True)
                angle_ij = torch.atan2(vec_ij[:,1], vec_ij[:,0]).unsqueeze(1)
                
                p_rydberg_i = p_rydberg[edge_index[0]]
                p_rydberg_j = p_rydberg[edge_index[1]]
                correlations = (p_rydberg_i * p_rydberg_j)

                edge_attr = torch.cat([
                    angle_ij,     # [E,1] geometric
                    correlations, # [E,1] experimental
                    dist_ij      # [E,1] normalized distances
                ], dim=1)

            # Calculate Mutual Information
            mutual_info = calculate_classical_mutual_information(
                p_rydberg.squeeze(),
                top_indices,
                top_probs,
                mask_tensor.squeeze(),
                N
            )

            # Skip if MI calculation was invalid
            if torch.isnan(mutual_info):
                skipped_count += 1
                continue

            # Create Data object
            target_vne = torch.tensor([row['Von_Neumann_Entropy']], dtype=torch.float)
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=target_vne,
                mutual_info=mutual_info
            )

            # Add remaining properties
            data.system_size = torch.tensor([[N]], dtype=torch.float)
            total_ryd = p_rydberg.sum()
            data.total_rydberg = total_ryd
            data.rydberg_density = total_ryd / N
            data.config_entropy = torch.tensor([[row['Von_Neumann_Entropy']]], dtype=torch.float)
            
            nA_val = float(mask_tensor.sum().item())
            nB_val = N - nA_val
            data.nA = torch.tensor([[nA_val]], dtype=torch.float)
            data.nB = torch.tensor([[nB_val]], dtype=torch.float)

            data_list.append(data)

            if (idx+1) % 20000 == 0:
                logging.info(f"Processed {idx+1} rows so far...")
                logging.info(f"Skipped {skipped_count} rows (prob < {CONFIG['prob_threshold']})")
                logging.info(f"Average probability coverage: {np.mean(prob_dist_stats):.4f}")

        # Final statistics
        logging.info(f"\nFinal statistics:")
        logging.info(f"Total samples processed: {total_count}")
        logging.info(f"Samples skipped: {skipped_count} ({100*skipped_count/total_count:.2f}%)")
        logging.info(f"Samples kept: {len(data_list)} ({100*len(data_list)/total_count:.2f}%)")
        logging.info(f"Average probability coverage: {np.mean(prob_dist_stats):.4f}")
        logging.info(f"Min probability coverage: {np.min(prob_dist_stats):.4f}")
        logging.info(f"Max probability coverage: {np.max(prob_dist_stats):.4f}")

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