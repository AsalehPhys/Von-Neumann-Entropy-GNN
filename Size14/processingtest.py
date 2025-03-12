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
from torch_geometric.data.data import DataEdgeAttr  # Add this import
import torch.serialization
# Add these safe globals
torch.serialization.add_safe_globals([Data, DataEdgeAttr])


CONFIG = {
    'data_paths': [
        'Rydbergsize10Delta2.5.parquet',
    ],
    'processed_dir': './processed_size10_2.5Delta_sym',
    'processed_file_name': 'data.pt',
    'random_seed': 42,
    'chunk_size': 800,  # Size of chunks for processing
    'use_gpu': False,  # GPU conversion happens at the end
    'save_interval': 50,  # Save intermediate results every 50 chunks
    'num_workers': 8,  # Number of CPU threads
    'distance_cutoff': 6,
    'recalculate_partitions': True  # Flag to create symmetric masks and recalculate VNE and MI
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
                
                entropy = -np.sum(positive_eigenvals * np.log(positive_eigenvals)) if len(positive_eigenvals) > 0 else 0
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

def create_symmetric_mask(Ny, Nx):
    """Create a symmetric subsystem mask based on the lattice dimensions."""
    N = Ny * Nx
    mask = np.zeros(N, dtype=np.float32)
    
    # For even Ny, partition like:
    # 0 0 1 1
    # 0 0 1 1
    if Ny % 2 == 0:
        half_Ny = Ny // 2
        for row in range(Nx):
            for col in range(Ny):
                idx = row * Ny + col
                if col >= half_Ny:  # Right half is subsystem A (1)
                    mask[idx] = 1
    # For odd Ny, partition like:
    # 0 0 1
    # 0 1 1
    else:
        half_Ny = Ny // 2
        for row in range(Nx):
            for col in range(Ny):
                idx = row * Ny + col
                if (row == 0 and col > half_Ny) or (row > 0 and col >= half_Ny):
                    mask[idx] = 1
    
    return mask.reshape(-1, 1)

def calculate_classical_MI(psi_gs, A_indices, N):
    """Calculate classical mutual information using the second code's convention.
    
    Args:
        psi_gs: The full quantum state vector
        A_indices: Indices of subsystem A
        N: Total number of spins/qubits
        
    Returns:
        float: The classical mutual information
    """
    # Get subsystem B indices
    B_indices = [i for i in range(N) if i not in A_indices]
    
    # Calculate joint probabilities
    p_joint = np.abs(psi_gs) ** 2
    
    # Calculate marginal probabilities
    p_A = np.zeros(2**len(A_indices))
    p_B = np.zeros(2**len(B_indices))
    
    for state_idx in range(2**N):
        if p_joint[state_idx] < 1e-12:  # Skip negligible probabilities
            continue
            
        # Use the same bit ordering convention as the second code
        state_bin = format(state_idx, f'0{N}b')
        A_state = int(''.join(state_bin[i] for i in A_indices), 2)
        B_state = int(''.join(state_bin[i] for i in B_indices), 2)
        
        p_A[A_state] += p_joint[state_idx]
        p_B[B_state] += p_joint[state_idx]
    
    # Calculate classical mutual information (still using log2 for bits)
    MI = 0.0
    for state_idx in range(2**N):
        if p_joint[state_idx] < 1e-12:  # Skip negligible probabilities
            continue
            
        # Use the same bit ordering convention as the second code
        state_bin = format(state_idx, f'0{N}b')
        A_state = int(''.join(state_bin[i] for i in A_indices), 2)
        B_state = int(''.join(state_bin[i] for i in B_indices), 2)
        
        # Skip if any marginal probability is zero
        if p_A[A_state] < 1e-12 or p_B[B_state] < 1e-12:
            continue
            
        MI += p_joint[state_idx] * np.log(p_joint[state_idx] / (p_A[A_state] * p_B[B_state]))
    
    return float(MI)

def reshape_for_subsystem(psi, A_indices, N):
    """Reshape wavefunction for bipartition using consistent bit ordering.
    
    This version ensures consistency with the second code's bit ordering convention.
    """
    A_indices = sorted(A_indices)
    B_indices = [i for i in range(N) if i not in A_indices]
    N_A = len(A_indices)
    N_B = N - N_A

    psi_reshaped = np.zeros((2**N_A, 2**N_B), dtype=psi.dtype)

    A_pos_map = {spin: pos for pos, spin in enumerate(A_indices)}
    B_pos_map = {spin: pos for pos, spin in enumerate(B_indices)}

    for i in range(2**N):
        # Skip negligible amplitudes for efficiency
        if abs(psi[i]) < 1e-12:  
            continue
        
        # Extract bits using the new convention (direct indexing)
        state_bin = format(i, f'0{N}b')
        
        # Map bits to subsystem states (consistent with second code)
        bits_A = ''.join(state_bin[i] for i in A_indices)
        bits_B = ''.join(state_bin[i] for i in B_indices)
        
        # Convert binary strings to integers
        i_A = int(bits_A, 2)
        i_B = int(bits_B, 2)

        psi_reshaped[i_A, i_B] = psi[i]

    return psi_reshaped

def calculate_von_neumann_entropy(psi, A_indices, N):
    """Calculate von Neumann entropy for a given subsystem partition.
    
    Uses the adjusted reshape_for_subsystem function with consistent bit ordering.
    
    Args:
        psi: The full quantum state vector
        A_indices: Indices of subsystem A
        N: Total number of spins/qubits
        
    Returns:
        float: The von Neumann entropy in bits (using log2)
    """
    # Reshape for subsystem calculation with consistent ordering
    psi_reshaped = reshape_for_subsystem(psi, A_indices, N)
    
    # Singular value decomposition
    U, s, Vh = np.linalg.svd(psi_reshaped, full_matrices=False)
    
    # Calculate entropy from singular values
    s_squared = s**2
    s_squared_normalized = s_squared / np.sum(s_squared)
    entropy = -np.sum(s_squared_normalized * np.log(s_squared_normalized + 1e-12))
    
    return entropy

# Usage in the process_single_row function would remain the same,
# but now these functions use consistent bit ordering with the second code

def process_single_row(row_data):
    try:
        Ny = row_data['Ny']
        Nx = 2
        N = Ny * Nx

        positions = np.array([(col * 1, row * 2)  
                            for row in range(Nx) 
                            for col in range(Ny)], dtype=np.float32)

        # Get state indices and probabilities
        state_indices = row_data['All_Indices']
        state_probs = row_data['All_Probabilities']
        
        # Calculate correlation matrix
        correlation_matrix = calculate_quantum_correlations_optimized(
            state_indices, state_probs, N)
        
        if CONFIG['recalculate_partitions']:
            # Create symmetric mask based on lattice dimensions
            mask = create_symmetric_mask(Ny, Nx)
            
            # Calculate von Neumann entropy and mutual information using the state vector approach
            # Convert probabilities to state vector
            psi = np.zeros(2**N)
            for idx, prob in zip(state_indices, state_probs):
                psi[idx] = np.sqrt(prob)
                
            # Normalize the state vector if needed
            psi = psi / np.linalg.norm(psi)
            
            # Get subsystem A indices from mask
            A_indices = np.where(mask.flatten() > 0)[0]
            
            # Calculate von Neumann entropy
            von_neumann_entropy = calculate_von_neumann_entropy(psi, A_indices, N)
            
            # Calculate classical mutual information
            classical_mi = calculate_classical_MI(psi, A_indices, N)
        else:
            # Use original mask and entropy values from the data
            mask = np.array([int(bit) for bit in row_data['Subsystem_Mask']], 
                          dtype=np.float32).reshape(-1, 1)
            von_neumann_entropy = row_data['Von_Neumann_Entropy']
            classical_mi = row_data['Classical_MI']

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

        # Use recalculated values or original values based on the flag
        if CONFIG['recalculate_partitions']:
            target_vne = np.array([von_neumann_entropy], dtype=np.float32)
            classical_mi_value = np.array([classical_mi], dtype=np.float32)
        else:
            target_vne = np.array([row_data['Von_Neumann_Entropy']], dtype=np.float32)
            classical_mi_value = np.array([row_data['Classical_MI']], dtype=np.float32)

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'target_vne': target_vne,
            'Delta_over_Omega': np.array([row_data['Delta_over_Omega']], dtype=np.float32),
            'Rb_over_a': np.array([row_data['Rb_over_a']], dtype=np.float32),
            'Energy': np.array([row_data['Energy']], dtype=np.float32),
            'Classical_MI': classical_mi_value,
            'system_size': np.array([[N]], dtype=np.float32),
            'nA': np.array([[float(mask.sum())]], dtype=np.float32),
            'nB': np.array([[float(N - mask.sum())]], dtype=np.float32)
        }
    except Exception as e:
        logging.error(f"Error processing row: {str(e)}")
        return None

def process_batch(batch_data):
    """Process a batch of data."""
    results = []
    for _, row in batch_data.iterrows():
        try:
            data = process_single_row(row)
            if data is not None:
                results.append(data)
        except Exception as e:
            logging.warning(f"Failed to process row: {str(e)}")
    return results

class ExperimentalSpinSystemDataset(InMemoryDataset):
    def __init__(self, root='.', transform=None, pre_transform=None):
        self.file_paths = CONFIG['data_paths']
        super().__init__(root, transform, pre_transform)
        
        # Check if processed file exists
        if os.path.exists(self.processed_paths[0]):
            try:
                # Try to load with the new weights_only=False for compatibility
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
                logging.info("Successfully loaded existing dataset with weights_only=False")
            except Exception as e:
                logging.warning(f"Could not load with weights_only=False: {str(e)}")
                logging.info("Processing data from scratch...")
                self.data, self.slices = self.process()
        else:
            self.data, self.slices = self.process()

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
        """Process the dataset, one small batch at a time."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.processed_dir, exist_ok=True)
            
            processed_chunks = []
            total_processed = 0
            start_time = time.time()
            last_save = start_time
            batch_counter = 0
            
            # Process each file
            for file_idx, file_path in enumerate(self.file_paths):
                logging.info(f"Processing file {file_idx + 1}/{len(self.file_paths)}: {file_path}")
                
                # Get the file size for reference
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logging.info(f"File size: {file_size_mb:.2f} MB")
                
                # Open parquet file using PyArrow - this doesn't load data into memory yet
                parquet_file = pq.ParquetFile(file_path)
                
                # Get the number of row groups in the file
                num_row_groups = parquet_file.metadata.num_row_groups
                logging.info(f"File contains {num_row_groups} row groups")
                
                # Process one row group at a time
                for row_group_idx in range(num_row_groups):
                    batch_start_time = time.time()
                    
                    try:
                        # Read only one row group at a time
                        logging.info(f"Reading row group {row_group_idx + 1}/{num_row_groups}")
                        
                        # This only loads one row group into memory
                        row_group = parquet_file.read_row_group(row_group_idx)
                        batch_data = row_group.to_pandas()
                        
                        # Log the memory usage
                        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
                        logging.info(f"Memory usage after reading row group: {memory_usage:.2f} MB")
                        
                        # Shuffle the batch
                        batch_data = batch_data.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
                        
                        # Process in smaller chunks to avoid memory spikes
                        num_chunks = (len(batch_data) + CONFIG['chunk_size'] - 1) // CONFIG['chunk_size']
                        
                        for chunk_idx in range(num_chunks):
                            chunk_start = chunk_idx * CONFIG['chunk_size']
                            chunk_end = min(chunk_start + CONFIG['chunk_size'], len(batch_data))
                            chunk_data = batch_data.iloc[chunk_start:chunk_end]
                            
                            # Process the chunk
                            logging.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} from row group {row_group_idx + 1}")
                            chunk_results = process_batch(chunk_data)
                            
                            if chunk_results:
                                processed_chunks.extend(chunk_results)
                                total_processed += len(chunk_results)
                                
                                logging.info(f"Processed {len(chunk_results)} rows successfully (Total: {total_processed})")
                                
                                # Save results periodically
                                current_time = time.time()
                                if current_time - last_save > 600 or batch_counter % CONFIG['save_interval'] == 0:
                                    temp_save_path = os.path.join(
                                        self.processed_dir, 
                                        f'temp_chunk_{batch_counter}.npy'
                                    )
                                    np.save(temp_save_path, processed_chunks)
                                    
                                    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
                                    logging.info(f"Saved intermediate results. Memory usage: {memory_usage:.2f} MB")
                                    last_save = current_time
                            
                            # Force garbage collection
                            del chunk_data
                            gc.collect()
                        
                        # Free memory and enforce garbage collection
                        del batch_data
                        del row_group
                        gc.collect()
                        
                        batch_counter += 1
                        batch_time = time.time() - batch_start_time
                        logging.info(f"Completed row group {row_group_idx + 1} in {batch_time:.2f}s")
                        
                    except Exception as e:
                        logging.error(f"Error processing row group {row_group_idx + 1}: {str(e)}")
                
                # Close the parquet file to free resources
                del parquet_file
                gc.collect()
                
                logging.info(f"Completed processing file {file_path}")
            
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
                if temp_file.startswith('temp_chunk_'):
                    try:
                        os.remove(os.path.join(self.processed_dir, temp_file))
                    except Exception as e:
                        logging.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
            
            data, slices = self.collate(data_list)
            
            # Save the processed data in a way that's compatible with future PyTorch versions
            try:
                torch.save((data, slices), self.processed_paths[0], _use_new_zipfile_serialization=True)
                logging.info(f"Saved processed data to {self.processed_paths[0]}")
            except TypeError:
                # Fallback for older PyTorch versions
                torch.save((data, slices), self.processed_paths[0])
                logging.info(f"Saved processed data to {self.processed_paths[0]} (old format)")
            
            total_time = time.time() - start_time
            logging.info(f"Total processing time: {total_time:.2f}s")
            logging.info(f"Processed {total_processed} rows successfully")
            
            return data, slices
            
        except Exception as e:
            logging.error(f"Error in process method: {str(e)}")
            raise

def load_data():
    # Check if data files exist
    for path in CONFIG['data_paths']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")
    
    # Initialize the dataset which will handle loading and processing the data
    return ExperimentalSpinSystemDataset(root=CONFIG['processed_dir'])

def main():
    setup_logging()
    set_seed(CONFIG['random_seed'])
    
    # Set number of CPU threads for PyTorch
    torch.set_num_threads(CONFIG['num_workers'])
    
    # Log whether we're using original or recalculated partitions
    if CONFIG['recalculate_partitions']:
        logging.info("USING SYMMETRIC PARTITIONS: Recalculating Von Neumann entropy and Mutual Information")
        logging.info("For even rungs, partition: 0 0 1 1 / 0 0 1 1")
        logging.info("For odd rungs, partition: 0 0 1 / 0 1 1")
    else:
        logging.info("Using original subsystem partitions from the parquet file")
    
    try:
        dataset = load_data()
        logging.info(f"Finished processing. Dataset length: {len(dataset)}")
        logging.info(f"Sample data object: {dataset[0]}")
    except Exception as e:
        logging.error(f"Failed to process dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()