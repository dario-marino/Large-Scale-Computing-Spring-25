#!/usr/bin/env python3
"""
Compute pairwise cosine similarity between patent embeddings using GPU acceleration.
Handles large datasets (8M+ patents) with GPU memory efficiency.
Modified to work with the existing environment setup by avoiding type annotation issues.
"""

import os
import time
import math
import gc
import numpy as np
import pandas as pd
# First try importing GPU libraries with error handling
try:
    # Try importing GPU libraries
    import cupy as cp
    import dask
    from dask.distributed import Client, wait
    from dask_cuda import LocalCUDACluster
    import dask.dataframe as dd
    import dask.array as da
    from dask_cuda.utils import get_n_gpus
    import cudf
    import rmm
    from cuml.metrics import pairwise_distances
    GPU_AVAILABLE = True
    print("GPU libraries loaded successfully")
except (ImportError, TypeError) as e:
    # If import fails, set flag to use CPU-based fallback
    print(f"GPU libraries failed to load: {e}")
    print("Falling back to CPU implementation")
    GPU_AVAILABLE = False
    
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
EMBEDDING_FILE = "/home/dariomarino/Downloads/patent_embeddings.npy"
PATENT_IDS_FILE = "/home/dariomarino/Downloads/patent_ids.csv"
OUTPUT_DIR = "/scratch/midway3/dariomarino/similarity_output_gpu"
BLOCK_SIZE = 2000  # Adjust based on available memory
SAVE_THRESHOLD = 5_000_000  
GPU_MEMORY_LIMIT = 0.8  # Use 80% of available GPU memory


def setup_dask_cluster():
    """Initialize DASK CUDA cluster optimized for this workload"""
    if not GPU_AVAILABLE:
        print("GPU libraries not available, skipping cluster setup")
        return None
        
    n_gpus = get_n_gpus()
    print(f"Found {n_gpus} GPUs")
    
    # Set up RMM pool allocator
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=None,  # Default is 1/2 of available memory
        managed_memory=False
    )
    
    # Create local CUDA cluster
    cluster = LocalCUDACluster(
        n_workers=n_gpus,
        threads_per_worker=1,
        rmm_pool_size=f"{int(GPU_MEMORY_LIMIT * 100)}%",
        # Enable spilling to host memory if needed
        device_memory_limit="70%",
        memory_limit="0.8",  # 80% of system memory
        # Enable NVLink for multi-GPU systems
        enable_nvlink=True,
        # Maximize locality of data
        protocol="ucx",
    )
    
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")
    return client


def calculate_block_similarities_gpu(block_i_idx, block_j_idx, embeddings, patent_ids, num_patents, block_size):
    """Calculate similarities between two blocks of embeddings using GPU"""
    # Get block ranges
    i_start = block_i_idx * block_size
    i_end = min(i_start + block_size, num_patents)
    j_start = block_j_idx * block_size
    j_end = min(j_start + block_size, num_patents)
    
    # Get patent IDs for these blocks
    ids_i = patent_ids[i_start:i_end]
    ids_j = patent_ids[j_start:j_end] if block_i_idx != block_j_idx else ids_i
    
    # Load embeddings into GPU memory
    try:
        # Move data to GPU
        block_i_embed = cp.asarray(embeddings[i_start:i_end], dtype=cp.float32)
        
        # Only load block_j if different from block_i
        if block_i_idx != block_j_idx:
            block_j_embed = cp.asarray(embeddings[j_start:j_end], dtype=cp.float32)
        else:
            block_j_embed = block_i_embed
            
        # Calculate pairwise cosine distances in one GPU operation
        # cuML returns distances, so we convert to similarities (1 - distance)
        cosine_distances = pairwise_distances(
            block_i_embed, 
            block_j_embed,
            metric='cosine'
        )
        
        similarities = 1.0 - cosine_distances
        
        # Convert back to CPU for processing results
        similarities_cpu = similarities.get()
        
        results = []
        
        # Process results
        for idx_i in range(len(ids_i)):
            # If same block, start from idx_i+1 to avoid duplicates and self-comparisons
            # For different blocks, compute all pairs
            j_range_start = idx_i + 1 if block_i_idx == block_j_idx else 0
            
            for idx_j in range(j_range_start, len(ids_j)):
                id_i = ids_i[idx_i]
                id_j = ids_j[idx_j]
                
                # Skip self-comparisons when in different blocks
                if id_i == id_j:
                    continue
                
                sim = float(similarities_cpu[idx_i, idx_j])
                
                # Ensure consistent ordering of patent IDs
                if id_i < id_j:
                    results.append((id_i, id_j, sim))
                else:
                    results.append((id_j, id_i, sim))
        
        # Free GPU memory explicitly
        del block_i_embed
        del similarities
        if block_i_idx != block_j_idx:
            del block_j_embed
        cp.get_default_memory_pool().free_all_blocks()
        
        return results
    
    except Exception as e:
        print(f"Error processing blocks {block_i_idx},{block_j_idx}: {e}")
        # Return empty list on error
        return []


def calculate_block_similarities_cpu(block_i_idx, block_j_idx, embeddings, patent_ids, num_patents, block_size):
    """Calculate similarities between two blocks of embeddings using CPU (fallback)"""
    # Get block ranges
    i_start = block_i_idx * block_size
    i_end = min(i_start + block_size, num_patents)
    j_start = block_j_idx * block_size
    j_end = min(j_start + block_size, num_patents)
    
    # Get patent IDs for these blocks
    ids_i = patent_ids[i_start:i_end]
    ids_j = patent_ids[j_start:j_end] if block_i_idx != block_j_idx else ids_i
    
    # Load embeddings
    block_i_embed = embeddings[i_start:i_end]
    
    # Only load block_j if different from block_i
    if block_i_idx != block_j_idx:
        block_j_embed = embeddings[j_start:j_end]
    else:
        block_j_embed = block_i_embed
        
    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(block_i_embed, block_j_embed)
    
    results = []
    
    # Process results
    for idx_i in range(len(ids_i)):
        # If same block, start from idx_i+1 to avoid duplicates and self-comparisons
        # For different blocks, compute all pairs
        j_range_start = idx_i + 1 if block_i_idx == block_j_idx else 0
        
        for idx_j in range(j_range_start, len(ids_j)):
            id_i = ids_i[idx_i]
            id_j = ids_j[idx_j]
            
            # Skip self-comparisons when in different blocks
            if id_i == id_j:
                continue
            
            sim = float(similarities[idx_i, idx_j])
            
            # Ensure consistent ordering of patent IDs
            if id_i < id_j:
                results.append((id_i, id_j, sim))
            else:
                results.append((id_j, id_i, sim))
    
    # Force garbage collection
    del block_i_embed
    if block_i_idx != block_j_idx:
        del block_j_embed
    del similarities
    gc.collect()
    
    return results


def save_results(results, output_dir, part_id):
    """Save results to parquet file"""
    if not results:
        return
    
    # Create DataFrame and save to parquet
    schema = pa.schema([
        ('patent_id_1', pa.string()),
        ('patent_id_2', pa.string()),
        ('similarity', pa.float32())
    ])
    
    df = pd.DataFrame(results, columns=['patent_id_1', 'patent_id_2', 'similarity'])
    table = pa.Table.from_pandas(df, schema=schema)
    
    output_file = os.path.join(output_dir, f"similarities_part{part_id}.parquet")
    pq.write_table(table, output_file)
    
    print(f"Saved {len(results)} similarities to {output_file}")
    return output_file


def main():
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set up client if GPU is available
    client = None
    if GPU_AVAILABLE:
        client = setup_dask_cluster()
        print(f"Client: {client}")
    else:
        print("Running in CPU mode (no GPU acceleration)")
    
    print(f"Loading patent IDs from {PATENT_IDS_FILE}")
    # Load patent IDs (keeping on CPU)
    patent_ids = pd.read_csv(PATENT_IDS_FILE, dtype=str, low_memory=False).iloc[:, 0].tolist()
    patent_ids = [str(pid) for pid in patent_ids]
    num_patents = len(patent_ids)
    print(f"Loaded {num_patents} patent IDs")
    
    print(f"Loading embeddings from {EMBEDDING_FILE}")
    # Memory-map embeddings file for efficient access
    embeddings = np.load(EMBEDDING_FILE, mmap_mode='r')
    print(f"Embedding shape: {embeddings.shape}")
    
    # Calculate number of blocks
    block_size = BLOCK_SIZE
    num_blocks = math.ceil(num_patents / block_size)
    print(f"Using block size: {block_size}, resulting in {num_blocks} blocks")
    
    # Generate all block pairs (upper triangular matrix including diagonal)
    block_pairs = []
    for i in range(num_blocks):
        for j in range(i, num_blocks):
            block_pairs.append((i, j))
    
    total_block_pairs = len(block_pairs)
    print(f"Total block pairs to process: {total_block_pairs}")
    
    # Process blocks in batches
    batch_size = 16 if GPU_AVAILABLE else 4  # Process this many block pairs in parallel
    num_batches = math.ceil(total_block_pairs / batch_size)
    
    file_counter = 0
    all_futures = []
    
    # Choose the appropriate function based on GPU availability
    calculation_function = calculate_block_similarities_gpu if GPU_AVAILABLE else calculate_block_similarities_cpu
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_block_pairs)
        current_batch = block_pairs[batch_start:batch_end]
        
        print(f"Processing batch {batch_idx+1}/{num_batches} with {len(current_batch)} block pairs")
        
        if GPU_AVAILABLE and client:
            # Submit tasks to cluster
            futures = []
            for block_i, block_j in current_batch:
                future = client.submit(
                    calculation_function,
                    block_i, block_j, embeddings, patent_ids, num_patents, block_size
                )
                futures.append(future)
            
            # Wait for current batch to complete
            results = client.gather(futures)
            
            # Flatten results
            all_results = []
            for result_list in results:
                all_results.extend(result_list)
        else:
            # Process in sequence on CPU
            all_results = []
            for block_i, block_j in current_batch:
                result_list = calculation_function(
                    block_i, block_j, embeddings, patent_ids, num_patents, block_size
                )
                all_results.extend(result_list)
                
        # Save if enough results or last batch
        if len(all_results) >= SAVE_THRESHOLD or batch_idx == num_batches - 1:
            save_file = save_results(all_results, OUTPUT_DIR, file_counter)
            file_counter += 1
            all_results = []
            
        # Force garbage collection
        gc.collect()
    
    # Wait for all tasks to complete if using GPU
    if GPU_AVAILABLE and client:
        print("Waiting for all remaining tasks to complete...")
        wait(all_futures)
        
        # Close client
        client.close()
    
    total_time = time.time() - start_time
    print(f"Processing completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("All processes completed.")


if __name__ == "__main__":
    main()