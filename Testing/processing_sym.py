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
import torch.serialization
torch.serialization.add_safe_globals([Data])


CONFIG = {
    'data_paths': [
        'Rydbergsize9Delta2.5.parquet',
    ],
    'processed_dir': './processed_Delta2.5_size9_r6',
    'processed_file_name': 'data.pt',
    'random_seed': 42,
    'chunk_size': 400,  # Size of chunks for sequential processing
    'use_gpu': False,  # GPU conversion happens at the end
    'save_interval': 50,  # Save intermediate results every 50 chunks
    'num_workers': 1,  # No parallel processing
    'batch_size': 1,  # Process 1 chunk at a time
    'timeout': 600,  # 10-minute timeout per batch
    'distance_cutoff': 6
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

def reshape_for_subsystem(psi, A_indices, N):
    """Reshape wavefunction for bipartition"""
    A_indices = sorted(A_indices)
    B_indices = [i for i in range(N) if i not in A_indices]
    N_A = len(A_indices)
    N_B = N - N_A

    psi_reshaped = np.zeros((2**N_A, 2**N_B), dtype=psi.dtype)

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

def calculate_vn_entropy(psi, A_indices, N):
    """Calculate von Neumann entropy for a given subsystem partition"""
    psi_reshaped = reshape_for_subsystem(psi, A_indices, N)
    
    U, s, Vh = np.linalg.svd(psi_reshaped, full_matrices=False)
    s_squared = s**2
    s_squared_normalized = s_squared / np.sum(s_squared)
    entropy = -np.sum(s_squared_normalized * np.log(s_squared_normalized + 1e-12))
    
    return float(entropy)

def calculate_classical_MI(psi_gs, A_indices, N):
    # Get subsystem B indices
    B_indices = [i for i in range(N) if i not in A_indices]
    
    # Calculate joint probabilities
    p_joint = np.abs(psi_gs) ** 2
    
    # Calculate marginal probabilities
    p_A = np.zeros(2**len(A_indices))
    p_B = np.zeros(2**len(B_indices))
    
    for state_idx in range(2**N):
        state_bin = format(state_idx, f'0{N}b')
        A_state = int(''.join(state_bin[i] for i in A_indices), 2)
        B_state = int(''.join(state_bin[i] for i in B_indices), 2)
        p_A[A_state] += p_joint[state_idx]
        p_B[B_state] += p_joint[state_idx]
    
    # Calculate classical mutual information
    MI = 0
    for state_idx in range(2**N):
        if p_joint[state_idx] > 1e-12:  # Numerical stability
            state_bin = format(state_idx, f'0{N}b')
            A_state = int(''.join(state_bin[i] for i in A_indices), 2)
            B_state = int(''.join(state_bin[i] for i in B_indices), 2)
            MI += p_joint[state_idx] * np.log(p_joint[state_idx] / (p_A[A_state] * p_B[B_state] + 1e-12))
    
    return float(MI)

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

def create_symmetrical_partition(positions, N, Ny):
    """Create a symmetrical left/right partition for a 2×Ny system
    
    For odd Ny, the middle column is split between subsystems A and B:
    Example for Ny=5:
    1 1 1 0 0  (top row)
    1 1 0 0 0  (bottom row)
    """
    # Calculate boundary column for even/odd case
    if Ny % 2 == 0:  # Even case
        boundary_col = Ny // 2
        A_indices = []
        for i in range(N):
            col = int(round(positions[i, 0]))  # x-coordinate is the column
            if col < boundary_col:
                A_indices.append(i)
    else:  # Odd case
        middle_col = Ny // 2  # Integer division gives the middle column
        A_indices = []
        for i in range(N):
            col = int(round(positions[i, 0]))  # x-coordinate is the column
            row = int(round(positions[i, 1] / 2))  # y-coordinate/2 is the row (0 or 1)
            
            # Add to A if:
            # - column is less than the middle column, or
            # - it's in the middle column AND in the top row
            if col < middle_col or (col == middle_col and row == 0):
                A_indices.append(i)
    
    # Create the mask
    mask = np.zeros(N, dtype=np.int32)
    mask[A_indices] = 1
    
    return A_indices, mask

def process_single_row(row_data):
    try:
        Ny = row_data['Ny']
        Nx = 2
        N = Ny * Nx

        positions = np.array([(col * 1, row * 2)  
                            for row in range(Nx) 
                            for col in range(Ny)], dtype=np.float32)

        # Reconstruct the quantum state from indices and probabilities
        state_indices = row_data['All_Indices']
        state_probs = row_data['All_Probabilities']
        
        # Reconstruct the wavefunction
        psi = np.zeros(2**N, dtype=np.complex64)
        for idx, prob in zip(state_indices, state_probs):
            psi[idx] = np.sqrt(prob)  # For simplicity, assume phases are zero
        
        # Create symmetrical partition based on the lattice structure
        A_indices, mask = create_symmetrical_partition(positions, N, Ny)
        
        # Calculate von Neumann entropy with the symmetrical partition
        vn_entropy = calculate_vn_entropy(psi, A_indices, N)
        
        # Calculate classical mutual information with the symmetrical partition
        classical_mi = calculate_classical_MI(psi, A_indices, N)

        correlation_matrix = calculate_quantum_correlations_optimized(
            state_indices, state_probs, N)

        # Convert mask to the format needed for further processing
        mask = mask.reshape(-1, 1).astype(np.float32)

        # Calculate quantum and geometric features
        quantum_features = calculate_quantum_features(correlation_matrix, positions, N)
        geometric_features = calculate_geometric_features(positions, N)

        # Combine features with original node features
        node_features = np.concatenate([
            positions,
            correlation_matrix.diagonal().reshape(-1, 1),
            mask,
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

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'target_vne': np.array([vn_entropy], dtype=np.float32),  # Updated with new VN entropy
            'Delta_over_Omega': np.array([row_data['Delta_over_Omega']], dtype=np.float32),
            'Rb_over_a': np.array([row_data['Rb_over_a']], dtype=np.float32),
            'Energy': np.array([row_data['Energy']], dtype=np.float32),
            'Classical_MI': np.array([classical_mi], dtype=np.float32),  # Updated with new MI
            'system_size': np.array([[N]], dtype=np.float32),
            'nA': np.array([[float(len(A_indices))]], dtype=np.float32),  # Updated based on our partition
            'nB': np.array([[float(N - len(A_indices))]], dtype=np.float32)  # Updated based on our partition
        }
    except Exception as e:
        logging.error(f"Error processing row: {str(e)}")
        return None

def process_chunk(chunk_data):
    """Process a chunk of data."""
    results = []
    for _, row in chunk_data.iterrows():
        try:
            data = process_single_row(row)
            if data is not None:
                results.append(data)
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
            
            logging.info(f"Starting processing with {len(chunks)} chunks")
            
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
                        f"Processed {len(chunk_results)} rows "
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
            logging.info(f"Processed {total_processed} rows successfully")
            
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
        logging.info(f"Finished processing. Dataset length: {len(dataset)}")
        logging.info(f"Sample data object: {dataset[0]}")
    except Exception as e:
        logging.error(f"Failed to process dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()