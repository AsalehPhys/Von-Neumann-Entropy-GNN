import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import time
import logging
import random
import warnings
import gc
import psutil
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from torch_geometric.data import InMemoryDataset, Data
from itertools import combinations
# PyTorch versioning compatibility check
import torch
try:
    import torch.serialization
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([Data])
except (ImportError, AttributeError):
    # For newer PyTorch versions where add_safe_globals might not exist
    pass


CONFIG = {
    'data_paths': [
        'Rydbergsize9-10.parquet',
    ],
    'processed_dir': './processed_9-10_times5_r6',
    'processed_file_name': 'data.pt',
    'random_seed': 42,
    'chunk_size': 400,  # Size of chunks for sequential processing
    'use_gpu': False,  # GPU conversion happens at the end
    'save_interval': 50,  # Save intermediate results every 50 chunks
    'num_workers': 1,  # No parallel processing
    'batch_size': 1,  # Process 1 chunk at a time
    'timeout': 600,  # 10-minute timeout per batch
    'distance_cutoff': 6,
    'partitions_per_datapoint': 5  # Number of different partitions to create per datapoint
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
    if torch.cuda.is_available() and CONFIG['use_gpu']:
        torch.cuda.manual_seed_all(seed)

def calculate_quantum_correlations_optimized(state_indices, state_probs, N):
    """Optimized version using numpy operations."""
    correlations = np.zeros((N, N), dtype=np.float32)
    states = np.array(state_indices, dtype=np.int64)
    probs = np.array(state_probs, dtype=np.float32)
    
    # Vectorized operations
    for i in range(N):
        mask = (states & (1 << i)) != 0
        correlations[i, i] = np.sum(probs[mask])
    
    for i, j in combinations(range(N), 2):
        mask = ((states & (1 << i)) != 0) & ((states & (1 << j)) != 0)
        corr = np.sum(probs[mask])
        connected_corr = corr - correlations[i, i] * correlations[j, j]
        correlations[i, j] = connected_corr
        correlations[j, i] = connected_corr
    
    return correlations

def generate_random_partition(N):
    """Generate a random partition by creating a binary mask."""
    # Ensure we don't have trivial partitions (all 0s or all 1s)
    while True:
        mask = np.zeros(N, dtype=np.int64)
        num_in_subsystem = random.randint(1, N-1)  # At least 1, at most N-1
        subsystem_indices = random.sample(range(N), num_in_subsystem)
        mask[subsystem_indices] = 1
        
        # Check that we have at least one 0 and one 1
        if 0 < np.sum(mask) < N:
            return mask

def reshape_for_subsystem(psi, A_indices, N):
    """Reshape wavefunction for bipartition"""
    A_indices = sorted(A_indices)
    B_indices = [i for i in range(N) if i not in A_indices]
    N_A = len(A_indices)
    N_B = N - N_A

    psi_reshaped = np.zeros((2**N_A, 2**N_B), dtype=np.complex128)

    A_pos_map = {spin: pos for pos, spin in enumerate(A_indices)}
    B_pos_map = {spin: pos for pos, spin in enumerate(B_indices)}

    for i in range(2**N):
        i_bin = i
        i_A = 0
        i_B = 0
        for spin in range(N):
            bit = i_bin & 1
            i_bin >>= 1
            if spin in A_pos_map:
                i_A |= (bit << A_pos_map[spin])
            else:
                i_B |= (bit << B_pos_map[spin])

        psi_reshaped[i_A, i_B] = psi[i]

    return psi_reshaped

def calculate_von_neumann_entropy(probabilities, mask, N):
    """Calculate the von Neumann entropy for a given partition."""
    # Convert the mask to indices
    A_indices = np.where(mask == 1)[0]
    
    # Create a pseudo-wavefunction from the probability distribution
    # Note: This assumes real amplitudes with positive phase
    psi = np.sqrt(probabilities)
    
    # Reshape for the bipartition
    psi_reshaped = reshape_for_subsystem(psi, A_indices, N)
    
    # Calculate SVD and entropy
    U, s, Vh = np.linalg.svd(psi_reshaped, full_matrices=False)
    s_squared = s**2
    s_squared_normalized = s_squared / np.sum(s_squared)
    
    # Calculate the entropy (avoid log(0))
    entropy = -np.sum(s_squared_normalized * np.log2(s_squared_normalized + 1e-12))
    
    return entropy, mask, len(A_indices)

def calculate_classical_MI(probabilities, mask, N):
    """Calculate classical mutual information for a given partition."""
    # Identify A and B subsystems based on the mask
    A_indices = np.where(mask == 1)[0]
    B_indices = np.where(mask == 0)[0]
    
    # State indices from 0 to 2^N - 1
    state_indices = np.arange(2**N)
    
    # Calculate marginal probabilities
    p_A = np.zeros(2**len(A_indices))
    p_B = np.zeros(2**len(B_indices))
    
    for state_idx in range(2**N):
        # Extract the bits for subsystems A and B
        state_bits = [(state_idx >> i) & 1 for i in range(N)]
        A_state = sum(state_bits[i] << idx for idx, i in enumerate(A_indices))
        B_state = sum(state_bits[i] << idx for idx, i in enumerate(B_indices))
        
        p_A[A_state] += probabilities[state_idx]
        p_B[B_state] += probabilities[state_idx]
    
    # Calculate mutual information
    MI = 0.0
    for state_idx in range(2**N):
        if probabilities[state_idx] > 1e-12:  # Avoid numerical issues
            state_bits = [(state_idx >> i) & 1 for i in range(N)]
            A_state = sum(state_bits[i] << idx for idx, i in enumerate(A_indices))
            B_state = sum(state_bits[i] << idx for idx, i in enumerate(B_indices))
            
            joint_prob = probabilities[state_idx]
            A_prob = p_A[A_state]
            B_prob = p_B[B_state]
            
            MI += joint_prob * np.log2(joint_prob / (A_prob * B_prob + 1e-12))
    
    return MI
    
def calculate_quantum_features(correlation_matrix, positions, N):
    quantum_features = {}
    window_sizes = [2, 3, 4]
    
    for window_size in window_sizes:
        local_obs = []
        for i in range(N):
            window_indices = [j for j in range(N) if 
                            np.linalg.norm(positions[i] - positions[j]) <= CONFIG['distance_cutoff']]
            
            if len(window_indices) > 1:
                window_corr = correlation_matrix[np.ix_(window_indices, window_indices)]
                eigenvals = np.linalg.eigvalsh(window_corr)
                positive_eigenvals = eigenvals[eigenvals > 1e-10]
                
                entropy = -np.sum(positive_eigenvals * np.log2(positive_eigenvals)) if len(positive_eigenvals) > 0 else 0
                pr = (np.sum(positive_eigenvals)**2 / np.sum(positive_eigenvals**2)) if len(positive_eigenvals) > 0 else 1
                purity = np.trace(window_corr @ window_corr)
                fluct = np.trace(window_corr @ (np.eye(len(window_indices)) - window_corr))
            else:
                entropy, pr, purity, fluct = 0, 1, 1, 0
            
            local_obs.append([entropy, pr, purity, fluct])
        
        quantum_features[f'local_obs_w{window_size}'] = np.array(local_obs, dtype=np.float32)
    
    return quantum_features

def calculate_geometric_features(positions, N):
    normalized_positions = positions / np.sqrt(N)
    center = np.mean(normalized_positions, axis=0)
    max_coords = np.max(normalized_positions, axis=0)
    min_coords = np.min(normalized_positions, axis=0)
    
    geometric_features = []
    for pos in normalized_positions:
        dist_to_boundaries = np.concatenate([
            pos - min_coords,
            max_coords - pos
        ])
        
        radial_dist = np.linalg.norm(pos - center)
        rel_pos = pos - center
        angle = np.arctan2(rel_pos[1], rel_pos[0]) / (2 * np.pi)
        neighbor_dist = 1.5 / np.sqrt(N)
        num_neighbors = np.sum(
            np.linalg.norm(normalized_positions - pos, axis=1) < neighbor_dist
        ) - 1
        
        features = np.concatenate([
            dist_to_boundaries,
            [radial_dist, angle, num_neighbors / N]
        ])
        geometric_features.append(features)
    
    return np.array(geometric_features, dtype=np.float32)

def create_edges_with_cutoff(positions, N):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= CONFIG['distance_cutoff']:
                edges.append([i, j])
    return np.array(edges, dtype=np.int64).T

def process_single_row(row_data):
    try:
        Ny = row_data['Ny']
        Nx = 2
        N = Ny * Nx

        positions = np.array([(col * 1, row * 2)  
                            for row in range(Nx) 
                            for col in range(Ny)], dtype=np.float32)

        correlation_matrix = calculate_quantum_correlations_optimized(
            row_data['All_Indices'], row_data['All_Probabilities'], N)

        # Generate multiple different partitions for this datapoint
        all_results = []
        for _ in range(CONFIG['partitions_per_datapoint']):
            # Generate a random partition mask
            mask = generate_random_partition(N)
            mask_str = ''.join(str(bit) for bit in mask)
            
            # Calculate entropy and mutual information for this partition
            probabilities = np.array(row_data['All_Probabilities'], dtype=np.float32)
            vn_entropy, _, nA = calculate_von_neumann_entropy(probabilities, mask, N)
            classical_mi = calculate_classical_MI(probabilities, mask, N)
            
            # Calculate quantum and geometric features
            quantum_features = calculate_quantum_features(correlation_matrix, positions, N)
            geometric_features = calculate_geometric_features(positions, N)

            # Combine features with original node features
            node_features = np.concatenate([
                positions,
                correlation_matrix.diagonal().reshape(-1, 1),
                mask.reshape(-1, 1),
                geometric_features,
                quantum_features['local_obs_w2'],
                quantum_features['local_obs_w3'],
                quantum_features['local_obs_w4']
            ], axis=1)

            edge_index = create_edges_with_cutoff(positions, N)
            
            if edge_index.size > 0:
                pos_i = positions[edge_index[0]]
                pos_j = positions[edge_index[1]]
                vec_ij = pos_j - pos_i
                dist_ij = np.linalg.norm(vec_ij, axis=1, keepdims=True) / np.sqrt(N)
                angle_ij = np.arctan2(vec_ij[:,1], vec_ij[:,0]).reshape(-1, 1)
                corr_values = correlation_matrix[edge_index[0], edge_index[1]].reshape(-1, 1)
                edge_attr = np.concatenate([angle_ij, corr_values, dist_ij], axis=1)
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)
                edge_attr = np.zeros((0, 3), dtype=np.float32)

            result = {
                'node_features': node_features,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'target_vne': np.array([vn_entropy], dtype=np.float32),
                'Delta_over_Omega': np.array([row_data['Delta_over_Omega']], dtype=np.float32),
                'Rb_over_a': np.array([row_data['Rb_over_a']], dtype=np.float32),
                'Energy': np.array([row_data['Energy']], dtype=np.float32),
                'Classical_MI': np.array([classical_mi], dtype=np.float32),
                'system_size': np.array([[N]], dtype=np.float32),
                'nA': np.array([[float(nA)]], dtype=np.float32),
                'nB': np.array([[float(N - nA)]], dtype=np.float32)
            }
            
            all_results.append(result)
            
        return all_results
        
    except Exception as e:
        logging.error(f"Error processing row: {str(e)}")
        return None

def process_chunk(chunk_data):
    """Process a chunk of data."""
    results = []
    for _, row in chunk_data.iterrows():
        try:
            data_list = process_single_row(row)
            if data_list is not None:
                results.extend(data_list)
        except Exception as e:
            logging.warning(f"Failed to process row: {str(e)}")
    return results

class ExperimentalSpinSystemDataset(InMemoryDataset):
    def __init__(self, dataframe, root='.', transform=None, pre_transform=None):
        self.df = dataframe
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0]) if os.path.exists(self.processed_paths[0]) else self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [CONFIG['processed_file_name']]

    def download(self):
        pass

    def numpy_to_torch_data(self, numpy_data, device='cpu'):
        """Convert numpy data dictionary to PyTorch Geometric Data object."""
        try:
            data = Data(
                x=torch.from_numpy(numpy_data['node_features']).float(),
                edge_index=torch.from_numpy(numpy_data['edge_index']).long(),
                edge_attr=torch.from_numpy(numpy_data['edge_attr']).float(),
                y=torch.from_numpy(numpy_data['target_vne']).float(),
                delta_over_omega=torch.from_numpy(numpy_data['Delta_over_Omega']).float(),
                rb_over_a=torch.from_numpy(numpy_data['Rb_over_a']).float(),
                energy=torch.from_numpy(numpy_data['Energy']).float(),
                MI=torch.from_numpy(numpy_data['Classical_MI']).float(),
                system_size=torch.from_numpy(numpy_data['system_size']).float(),
                nA=torch.from_numpy(numpy_data['nA']).float(),
                nB=torch.from_numpy(numpy_data['nB']).float()
            )
            return data.to(device) if device != 'cpu' else data
        except Exception as e:
            logging.error(f"Error converting numpy to torch: {str(e)}")
            return None

    def process(self):
        """Process the dataset with sequential numpy-based computations."""
        try:
            num_chunks = len(self.df) // CONFIG['chunk_size'] + 1
            chunks = np.array_split(self.df, num_chunks)
            
            processed_chunks = []
            total_processed = 0
            start_time = time.time()
            last_save = start_time
            
            logging.info(f"Starting processing with {len(chunks)} chunks, generating {CONFIG['partitions_per_datapoint']} partitions per data point")
            
            # Process chunks sequentially
            for chunk_idx, chunk in enumerate(chunks):
                chunk_start = time.time()
                logging.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
                
                chunk_results = process_chunk(chunk)
                if chunk_results:
                    processed_chunks.extend(chunk_results)
                    total_processed += len(chunk_results)
                    
                    logging.info(
                        f"Chunk {chunk_idx + 1}/{len(chunks)} complete. "
                        f"Processed {len(chunk_results)} data points "
                        f"(Total: {total_processed})"
                    )
                    
                    current_time = time.time()
                    if (current_time - last_save > 600 or 
                        (chunk_idx + 1) % CONFIG['save_interval'] == 0):
                        temp_save_path = os.path.join(
                            self.processed_dir, 
                            f'temp_numpy_chunk_{chunk_idx+1}.npy'
                        )
                        np.save(temp_save_path, processed_chunks)
                        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                        logging.info(
                            f"Saved intermediate results at chunk {chunk_idx+1}. "
                            f"Memory usage: {memory_usage:.2f} MB"
                        )
                        last_save = current_time
                else:
                    logging.warning(f"Chunk {chunk_idx + 1} returned no results")
                
                chunk_time = time.time() - chunk_start
                logging.info(f"Chunk {chunk_idx + 1} processed in {chunk_time:.2f}s")
                
                # Force garbage collection
                gc.collect()
            
            logging.info("Processing complete. Converting to PyTorch Geometric format...")
            
            if not processed_chunks:
                raise RuntimeError("No data was successfully processed")
            
            # Convert to PyTorch Geometric format
            data_list = []
            for numpy_data in processed_chunks:
                torch_data = self.numpy_to_torch_data(numpy_data)
                if torch_data is not None:
                    data_list.append(torch_data)
            
            # Clean up temporary files
            for temp_file in os.listdir(self.processed_dir):
                if temp_file.startswith('temp_numpy_chunk_'):
                    try:
                        os.remove(os.path.join(self.processed_dir, temp_file))
                    except Exception as e:
                        logging.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
            
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            
            total_time = time.time() - start_time
            logging.info(f"Total processing time: {total_time:.2f}s")
            logging.info(f"Processed {total_processed} data points successfully (from {len(self.df)} original data points)")
            
            return data, slices
            
        except Exception as e:
            logging.error(f"Error in process method: {str(e)}")
            raise

def load_data():
    df_list = []
    for path in CONFIG['data_paths']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")
        table = pq.read_table(path)
        df_temp = table.to_pandas()
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    df_shuffled = df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    return ExperimentalSpinSystemDataset(dataframe=df_shuffled, root=CONFIG['processed_dir'])

def main():
    setup_logging()
    set_seed(CONFIG['random_seed'])
    
    # Set number of CPU threads for PyTorch
    torch.set_num_threads(CONFIG['num_workers'])
    
    try:
        dataset = load_data()
        logging.info(f"Finished processing. Dataset length: {len(dataset)} (approx. {CONFIG['partitions_per_datapoint']}x increase)")
        logging.info(f"Sample data object: {dataset[0]}")
    except Exception as e:
        logging.error(f"Failed to process dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()