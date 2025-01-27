import os
import sys
import time
import logging
import random
import warnings
import gc
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing as mp
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from torch_geometric.data import InMemoryDataset, Data
from itertools import combinations

# Modified Configuration
CONFIG = {
    'data_paths': [
        'Rydberg50k8-9.parquet',
        'Rydberg1.5M1-8.parquet',
    ],
    'processed_dir': './processed_experimentalc',
    'processed_file_name': 'data.pt',
    'random_seed': 42,
    'chunk_size': 400,  # Size of chunks for parallel processing
    'use_gpu': False,  # GPU conversion happens at the end
    'save_interval': 50,  # Save intermediate results every 50 chunks
    'num_workers': min(mp.cpu_count() - 1, 8),  # Limit to 8 workers max
    'batch_size': 5,  # Process 5 chunks at a time
    'timeout': 600  # 10-minute timeout per batch
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

def process_single_row(row_data):
    """Process a single row of data using numpy arrays."""
    try:
        Nx = row_data['Nx']
        Ny = 2
        N = Nx * Ny

        positions = np.array([(col, row_idx) 
                            for row_idx in range(Nx) 
                            for col in range(Ny)], dtype=np.float32)

        correlation_matrix = calculate_quantum_correlations_optimized(
            row_data['Top_Indices'], row_data['Top_Probabilities'], N)

        mask = np.array([int(bit) for bit in row_data['Subsystem_Mask']], 
                       dtype=np.float32).reshape(-1, 1)

        node_features = np.concatenate([
            positions,
            correlation_matrix.diagonal().reshape(-1, 1),
            mask
        ], axis=1)

        edges = np.array(list(combinations(range(N), 2)))
        edge_index = edges.T

        pos_i = positions[edge_index[0]]
        pos_j = positions[edge_index[1]]
        vec_ij = pos_j - pos_i
        dist_ij = np.linalg.norm(vec_ij, axis=1, keepdims=True) / np.sqrt(N)
        angle_ij = np.arctan2(vec_ij[:,1], vec_ij[:,0]).reshape(-1, 1)
        
        corr_values = correlation_matrix[edge_index[0], edge_index[1]].reshape(-1, 1)
        edge_attr = np.concatenate([angle_ij, corr_values, dist_ij], axis=1)

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'target_vne': np.array([row_data['Von_Neumann_Entropy']], dtype=np.float32),
            'system_size': np.array([[N]], dtype=np.float32),
            'nA': np.array([[float(mask.sum())]], dtype=np.float32),
            'nB': np.array([[float(N - mask.sum())]], dtype=np.float32)
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
                system_size=torch.from_numpy(numpy_data['system_size']).float(),
                nA=torch.from_numpy(numpy_data['nA']).float(),
                nB=torch.from_numpy(numpy_data['nB']).float()
            )
            return data.to(device) if device != 'cpu' else data
        except Exception as e:
            logging.error(f"Error converting numpy to torch: {str(e)}")
            return None

    def process_chunk_parallel(self, chunk_data):
        """Process a chunk of data in parallel."""
        return process_chunk(chunk_data)

    def process(self):
        """Process the dataset with parallel numpy-based computations."""
        try:
            num_chunks = len(self.df) // CONFIG['chunk_size'] + 1
            chunks = np.array_split(self.df, num_chunks)
            
            processed_chunks = []
            total_processed = 0
            start_time = time.time()
            last_save = start_time
            
            logging.info(f"Starting processing with {len(chunks)} chunks")
            
            # Process chunks in batches
            for batch_idx in range(0, len(chunks), CONFIG['batch_size']):
                batch_start = time.time()
                batch_chunks = chunks[batch_idx:batch_idx + CONFIG['batch_size']]
                logging.info(f"\nStarting batch {batch_idx//CONFIG['batch_size'] + 1}/{(len(chunks)-1)//CONFIG['batch_size'] + 1}")
                logging.info(f"Total rows processed so far: {total_processed}")
                
                # Process batch in parallel
                with ProcessPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
                    future_to_chunk = {
                        executor.submit(process_chunk, chunk): i 
                        for i, chunk in enumerate(batch_chunks)
                    }
                    
                    for future in as_completed(future_to_chunk):
                        chunk_idx = batch_idx + future_to_chunk[future]
                        try:
                            chunk_results = future.result(timeout=CONFIG['timeout'])
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
                                
                        except TimeoutError:
                            logging.error(f"Timeout processing chunk {chunk_idx + 1}")
                            continue
                        except Exception as e:
                            logging.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
                            continue
                
                batch_time = time.time() - batch_start
                logging.info(
                    f"Batch complete. Time: {batch_time:.2f}s. "
                    f"Average per chunk: {batch_time/len(batch_chunks):.2f}s"
                )
                
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