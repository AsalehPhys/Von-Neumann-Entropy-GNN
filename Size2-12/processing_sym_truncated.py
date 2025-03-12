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
import multiprocessing as mp
from functools import partial

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
        'Rydberg0.5M.parquet',
    ],
    'processed_dir': './processed_size6_0.5M_truncated',
    'processed_file_name': 'data.pt',
    'random_seed': 42,
    'chunk_size': 10000,  # Size of chunks for sequential processing
    'use_gpu': False,  # GPU conversion happens at the end
    'save_interval': 50,  # Save intermediate results every 50 chunks
    'num_workers': max(1, os.cpu_count() - 3),  # Use all but one CPU core
    'batch_size': 64,  # Increased batch size for better parallelism
    'timeout': 600,  # 10-minute timeout per batch
    'distance_cutoff': 6,
    'parallel_processing': True  # Enable parallel processing
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
    
    # Use np.linalg.svd with full_matrices=False for better performance
    U, s, Vh = np.linalg.svd(psi_reshaped, full_matrices=False)
    s_squared = s**2
    s_squared_normalized = s_squared / np.sum(s_squared)
    
    # Use faster vectorized operation with a small epsilon for numerical stability
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
    
    # Vectorize state calculation for efficiency
    state_bins = np.array([format(i, f'0{N}b') for i in range(2**N)])
    
    for state_idx in range(2**N):
        state_bin = state_bins[state_idx]
        A_state = int(''.join(state_bin[i] for i in A_indices), 2)
        B_state = int(''.join(state_bin[i] for i in B_indices), 2)
        p_A[A_state] += p_joint[state_idx]
        p_B[B_state] += p_joint[state_idx]
    
    # Calculate classical mutual information with vectorized operations
    MI = 0
    valid_indices = np.where(p_joint > 1e-12)[0]  # Only consider non-zero probabilities
    
    for state_idx in valid_indices:
        state_bin = state_bins[state_idx]
        A_state = int(''.join(state_bin[i] for i in A_indices), 2)
        B_state = int(''.join(state_bin[i] for i in B_indices), 2)
        MI += p_joint[state_idx] * np.log(p_joint[state_idx] / (p_A[A_state] * p_B[B_state] + 1e-12))
    
    return float(MI)

def calculate_quantum_correlations_optimized(state_indices, state_probs, N):
    """Optimized version using numpy operations."""
    correlations = np.zeros((N, N), dtype=np.float32)
    states = np.array(state_indices, dtype=np.int64)
    probs = np.array(state_probs, dtype=np.float32)
    
    # Calculate diagonal elements (optimized)
    for i in range(N):
        mask = (states & (1 << i)) != 0
        correlations[i, i] = np.sum(probs[mask])
    
    # Calculate off-diagonal elements in parallel using numpy operations
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
    
    # Precompute distance matrix for faster lookups
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    for window_size in window_sizes:
        local_obs = []
        for i in range(N):
            # Use vectorized operations to find window indices
            window_indices = np.where(distance_matrix[i] <= CONFIG['distance_cutoff'])[0]
            
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
    
    # Vectorized operations for all positions at once
    dist_to_min = normalized_positions - min_coords
    dist_to_max = max_coords - normalized_positions
    radial_dist = np.linalg.norm(normalized_positions - center, axis=1)
    rel_pos = normalized_positions - center
    angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0]) / (2 * np.pi)
    
    # Calculate neighbor counts using vectorized distance calculation
    neighbor_dist = 1.5 / np.sqrt(N)
    dist_matrix = np.linalg.norm(normalized_positions[:, np.newaxis, :] - 
                               normalized_positions[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dist_matrix, np.inf)  # Exclude self
    neighbor_counts = np.sum(dist_matrix < neighbor_dist, axis=1) / N
    
    # Combine all features efficiently
    geometric_features = np.hstack([
        dist_to_min,
        dist_to_max,
        radial_dist.reshape(-1, 1),
        angles.reshape(-1, 1),
        neighbor_counts.reshape(-1, 1)
    ])
    
    return geometric_features.astype(np.float32)

def create_edges_with_cutoff(positions, N):
    # Vectorized distance calculation
    i, j = np.triu_indices(N, k=1)
    dists = np.linalg.norm(positions[i] - positions[j], axis=1)
    valid_edges = np.where(dists <= CONFIG['distance_cutoff'])[0]
    
    if len(valid_edges) > 0:
        edges = np.stack([i[valid_edges], j[valid_edges]])
        return edges
    else:
        return np.zeros((2, 0), dtype=np.int64)

def create_symmetrical_partition(positions, N, Ny):
    """Create a symmetrical left/right partition for a 2×Ny system"""
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
        
        # First reconstruct the full wavefunction for von Neumann entropy calculation
        psi_full = np.zeros(2**N, dtype=np.complex64)
        for idx, prob in zip(state_indices, state_probs):
            psi_full[idx] = np.sqrt(prob)  # For simplicity, assume phases are zero
        
        # Create symmetrical partition based on the lattice structure
        A_indices, mask = create_symmetrical_partition(positions, N, Ny)
        
        # Calculate von Neumann entropy with the symmetrical partition using full wavefunction
        vn_entropy = calculate_vn_entropy(psi_full, A_indices, N)
        
        # Now filter out states with probability less than 10^-4 for further processing
        threshold = 1e-4
        filter_mask = np.array(state_probs) >= threshold
        filtered_indices = np.array(state_indices)[filter_mask]
        filtered_probs = np.array(state_probs)[filter_mask]
        
        # Normalize the remaining probabilities to sum to 1
        filtered_probs = filtered_probs / np.sum(filtered_probs)
        
        # Reconstruct the truncated wavefunction for classical MI calculation
        psi_truncated = np.zeros(2**N, dtype=np.complex64)
        for idx, prob in zip(filtered_indices, filtered_probs):
            psi_truncated[idx] = np.sqrt(prob)
            
        # Calculate classical mutual information with the truncated wavefunction
        classical_mi = calculate_classical_MI(psi_truncated, A_indices, N)

        # Use filtered indices and probabilities for correlation matrix calculation
        correlation_matrix = calculate_quantum_correlations_optimized(
            filtered_indices, filtered_probs, N)

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
            'target_vne': np.array([vn_entropy], dtype=np.float32),
            'Delta_over_Omega': np.array([row_data['Delta_over_Omega']], dtype=np.float32),
            'Rb_over_a': np.array([row_data['Rb_over_a']], dtype=np.float32),
            'Energy': np.array([row_data['Energy']], dtype=np.float32),
            'Classical_MI': np.array([classical_mi], dtype=np.float32),
            'system_size': np.array([[N]], dtype=np.float32),
            'nA': np.array([[float(len(A_indices))]], dtype=np.float32),
            'nB': np.array([[float(N - len(A_indices))]], dtype=np.float32)
        }
    except Exception as e:
        logging.error(f"Error processing row: {str(e)}")
        return None

def process_chunk(chunk_data, worker_id=None):
    """Process a chunk of data."""
    if worker_id is not None:
        logging.info(f"Worker {worker_id} processing chunk of size {len(chunk_data)}")
    
    results = []
    for _, row in chunk_data.iterrows():
        try:
            data = process_single_row(row)
            if data is not None:
                results.append(data)
        except Exception as e:
            logging.warning(f"Failed to process row: {str(e)}")
    return results

def parallel_process_chunk(df_chunk, num_workers):
    """Process a chunk in parallel using multiple workers"""
    if len(df_chunk) < num_workers:
        return process_chunk(df_chunk)
    
    # Split the chunk into smaller parts for each worker
    chunk_splits = np.array_split(df_chunk, num_workers)
    
    # Create a pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Process each split in parallel with worker_id
        results = pool.starmap(
            process_chunk,
            [(chunk, i) for i, chunk in enumerate(chunk_splits)]
        )
    
    # Combine results from all workers
    combined_results = []
    for worker_result in results:
        combined_results.extend(worker_result)
    
    return combined_results

def load_and_process_data_in_batches():
    """Load and process parquet files in smaller batches to reduce memory usage"""
    # Create directory if it doesn't exist
    if not os.path.exists(CONFIG['processed_dir']):
        os.makedirs(CONFIG['processed_dir'])
    
    # Placeholder for the dataset we're building
    data_list = []
    total_processed = 0
    
    for path in CONFIG['data_paths']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")
        
        logging.info(f"Processing file: {path}")
        
        # Get metadata to determine number of rows without loading everything
        parquet_file = pq.ParquetFile(path)
        num_rows = parquet_file.metadata.num_rows
        batch_size = min(CONFIG['chunk_size'] * 10, max(1000, num_rows // 20))  # Adjust batch size based on file size
        
        logging.info(f"File contains {num_rows} rows. Processing in batches of {batch_size}")
        
        # Process the file in batches
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            batch_start = time.time()
            
            # Convert batch to pandas DataFrame
            df_batch = batch.to_pandas()
            
            # Apply random seed for consistent shuffling if needed
            df_batch = df_batch.sample(frac=1, random_state=CONFIG['random_seed'] + batch_idx).reset_index(drop=True)
            
            logging.info(f"Batch {batch_idx+1}: Processing {len(df_batch)} rows (rows {batch_idx*batch_size} to {min((batch_idx+1)*batch_size, num_rows)-1})")
            
            # Process the batch
            if CONFIG['parallel_processing'] and CONFIG['num_workers'] > 1:
                batch_results = parallel_process_chunk(df_batch, CONFIG['num_workers'])
            else:
                batch_results = process_chunk(df_batch)
            
            if batch_results:
                # Convert numpy batch results to PyTorch Geometric format
                for numpy_data in batch_results:
                    torch_data = numpy_to_torch_data(numpy_data)
                    if torch_data is not None:
                        data_list.append(torch_data)
                
                total_processed += len(batch_results)
                logging.info(f"Batch {batch_idx+1} complete. Processed {len(batch_results)} rows (Total: {total_processed})")
                
                # Save intermediate results periodically
                if (batch_idx + 1) % 5 == 0:  # Save every 5 batches
                    temp_save_path = os.path.join(
                        CONFIG['processed_dir'], 
                        f'temp_data_{len(data_list)}.pt'
                    )
                    try:
                        # Save the current data_list to a temporary file
                        if data_list:
                            data, slices = collate_data(data_list)
                            torch.save((data, slices), temp_save_path)
                            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                            logging.info(f"Saved intermediate results. Memory usage: {memory_usage:.2f} MB")
                    except Exception as e:
                        logging.error(f"Error saving intermediate results: {str(e)}")
            
            batch_time = time.time() - batch_start
            logging.info(f"Batch {batch_idx+1} processed in {batch_time:.2f}s")
            
            # Collect garbage less frequently to avoid processing pauses
            if batch_idx % 3 == 0:
                gc.collect()
    
    # Collate all processed data
    if not data_list:
        raise RuntimeError("No data was successfully processed")
    
    logging.info(f"Processing complete. Collating {len(data_list)} data objects...")
    
    # Collate data and save
    data, slices = collate_data(data_list)
    torch.save((data, slices), os.path.join(CONFIG['processed_dir'], CONFIG['processed_file_name']))
    
    logging.info(f"Finished processing. Total processed: {total_processed}")
    
    # Load and return the dataset from the saved file
    return torch.load(os.path.join(CONFIG['processed_dir'], CONFIG['processed_file_name']))

def numpy_to_torch_data(numpy_data, device='cpu'):
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

def collate_data(data_list):
    """Collate a list of Data objects."""
    from torch_geometric.data import InMemoryDataset
    return InMemoryDataset.collate(data_list)

class ExperimentalSpinSystemDataset(InMemoryDataset):
    def __init__(self, root='.', transform=None, pre_transform=None):
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

    def process(self):
        """Process the dataset with batched loading of parquet files."""
        try:
            return load_and_process_data_in_batches()
        except Exception as e:
            logging.error(f"Error in process method: {str(e)}")
            raise

def main():
    setup_logging()
    set_seed(CONFIG['random_seed'])
    
    # Set PyTorch to use all available cores
    torch.set_num_threads(CONFIG['num_workers'])
    
    # Display system info
    logging.info(f"Number of CPU cores: {os.cpu_count()}")
    logging.info(f"Using {CONFIG['num_workers']} worker threads")
    logging.info(f"Parallel processing: {CONFIG['parallel_processing']}")
    
    try:
        # Create and process the dataset
        dataset = ExperimentalSpinSystemDataset(root=CONFIG['processed_dir'])
        logging.info(f"Finished processing. Dataset length: {len(dataset)}")
        logging.info(f"Sample data object: {dataset[0]}")
    except Exception as e:
        logging.error(f"Failed to process dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Enable multiprocessing for Windows
    mp.freeze_support()
    main()