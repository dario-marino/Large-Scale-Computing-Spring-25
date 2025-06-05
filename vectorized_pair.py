#!/usr/bin/env python3
"""
Compute pairwise cosine similarity between patent embeddings using MPI parallelization.
Handles large datasets (8M+ patents) with memory efficiency.
Optimized to avoid redundant calculations.
Enhanced with checkpointing and resume functionality.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mpi4py import MPI
import os
import time
from pathlib import Path
import math
from scipy.spatial.distance import cosine
import gc  # For garbage collection
# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configuration
EMBEDDING_FILE = "/home/dariomarino/Downloads/patent_embeddings.npy"
PATENT_IDS_FILE = "/home/dariomarino/Downloads/patent_ids.csv"
OUTPUT_DIR = "/scratch/midway3/dariomarino/similarity_output"
BLOCK_SIZE = 500                                 
SAVE_THRESHOLD = 1_000_000                       

# Resume configuration - set these if resuming from a previous run
# Copy the progress vector from your parsing script output here
RESUME_PROGRESS = [2541, 2311, 2551, 2551, 2551, 2551, 2551, 2551, 2551, 2551, 2501, 2501, 2721, 2501,
                   2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501,
                   2501, 2501, 2501, 2501, 2721, 2501, 2501, 2501, 2721, 2501, 2501, 2501, 2501, 2501,
                   2501, 2501, 2721, 2501, 2501, 2501, 2501, 2721, 2501, 2501, 2501, 2501, 2501, 2501,
                   2501, 2721, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2721, 2501, 2501, 2501,
                   2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2721,
                   2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501, 2501,
                   2501, 2501, 2681, 2451, 2681, 2461, 2451, 2681, 2451, 2681, 2461, 2681, 2451, 2451,
                   2681, 2451, 2451, 2451, 2451, 2451, 2681, 2681, 2461, 2461, 2461, 2461, 2451, 2451,
                   2451, 2451]

# Set to True if you want to resume from the progress above, False for fresh start
RESUME_FROM_CHECKPOINT = True

# Create output directory if it doesn't exist
if rank == 0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
comm.Barrier()

def print_progress_status(rank, current_pair, total_pairs, start_time):
    """Print detailed progress status."""
    elapsed = time.time() - start_time
    progress_pct = (current_pair / total_pairs) * 100 if total_pairs > 0 else 0
    
    print(f"Rank {rank}: Progress {current_pair}/{total_pairs} ({progress_pct:.2f}%) - "
          f"Elapsed: {elapsed/60:.1f}min")

# Function to calculate cosine similarity
def calculate_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return 1 - cosine(vec1, vec2)  # Convert distance to similarity

def main():
    start_time = time.time()
    
    if rank == 0:
        print(f"Starting patent similarity computation with {size} processes")
        if RESUME_FROM_CHECKPOINT:
            print("RESUMING from previous checkpoint...")
        print(f"Loading patent IDs from {PATENT_IDS_FILE}")
    
    # Fixed loading of patent IDs to handle mixed types
    patent_ids = pd.read_csv(PATENT_IDS_FILE, dtype=str, low_memory=False).iloc[:, 0].tolist()
    
    # Convert patent IDs to strings to avoid type issues with PyArrow
    patent_ids = [str(pid) for pid in patent_ids]
    
    num_patents = len(patent_ids)
    
    if rank == 0:
        print(f"Found {num_patents} patents")
        print(f"Loading embeddings from {EMBEDDING_FILE}")
    
    # Memory-map the embedding file instead of loading it all at once
    embeddings = np.load(EMBEDDING_FILE, mmap_mode='r')
    
    if rank == 0:
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Using block size: {BLOCK_SIZE}")
    
    # Calculate the total number of blocks
    num_blocks = math.ceil(num_patents / BLOCK_SIZE)
    
    # Only compute upper triangular matrix of block pairs (including diagonal)
    total_block_pairs = (num_blocks * (num_blocks + 1)) // 2
    
    if rank == 0:
        print(f"Total block pairs: {total_block_pairs}")
    
    # Distribute block pairs among processes and determine resume point
    block_pairs = []
    pair_idx = 0
    resume_from_pair = 0
    
    # If resuming, get the starting point for this rank
    if RESUME_FROM_CHECKPOINT and rank < len(RESUME_PROGRESS):
        resume_from_pair = RESUME_PROGRESS[rank]
        if rank == 0:
            print(f"Resuming from checkpoint. Rank progress: min={min(RESUME_PROGRESS)}, max={max(RESUME_PROGRESS)}")
    
    for i in range(num_blocks):
        for j in range(i, num_blocks):  # Only upper triangle including diagonal
            if pair_idx % size == rank:
                # Only add this pair if we haven't processed it yet (for resume)
                if pair_idx >= resume_from_pair:
                    block_pairs.append((i, j, pair_idx))  # Include original index for tracking
            pair_idx += 1
    
    total_pairs_for_rank = len(block_pairs)
    print(f"Rank {rank}: Will process {total_pairs_for_rank} block pairs "
          f"(resumed from pair {resume_from_pair})")
    
    # Create schema for output
    schema = pa.schema([
        ('patent_id_1', pa.string()),
        ('patent_id_2', pa.string()),
        ('similarity', pa.float32())
    ])
    
    results_count = 0
    file_counter = 0
    similarity_results = []
    processed_pairs_count = 0
    
    # Process assigned block pairs
    for block_idx, (block_i, block_j, original_pair_idx) in enumerate(block_pairs):
        
        # Print progress every 10 block pairs
        if block_idx % 10 == 0:
            print(f"Rank {rank}: Processing block pair {block_idx + 1}/{total_pairs_for_rank} "
                  f"({block_i},{block_j}) [Original index: {original_pair_idx}]")
            print_progress_status(rank, block_idx, total_pairs_for_rank, start_time)
        
        # Calculate start and end indices for both blocks
        i_start = block_i * BLOCK_SIZE
        i_end = min(i_start + BLOCK_SIZE, num_patents)
        j_start = block_j * BLOCK_SIZE
        j_end = min(j_start + BLOCK_SIZE, num_patents)
        
        # Load embeddings for these blocks
        block_i_embed = embeddings[i_start:i_end].astype(np.float32)
        # Only load block_j if it's different from block_i to save memory
        if block_i != block_j:
            block_j_embed = embeddings[j_start:j_end].astype(np.float32)
        else:
            block_j_embed = block_i_embed
        
        # Get corresponding patent IDs
        ids_i = patent_ids[i_start:i_end]
        ids_j = patent_ids[j_start:j_end] if block_i != block_j else ids_i
        
        # Calculate similarities for this block pair
        for idx_i, (id_i, embed_i) in enumerate(zip(ids_i, block_i_embed)):
            # For same block, start from idx_i+1 to avoid duplicates and self-comparisons
            # For different blocks, compute all pairs
            j_range_start = idx_i + 1 if block_i == block_j else 0
            
            for idx_j in range(j_range_start, len(block_j_embed)):
                id_j = ids_j[idx_j]
                
                # Skip self-comparisons when in different blocks
                if id_i == id_j:
                    continue
                
                embed_j = block_j_embed[idx_j]
                sim = calculate_similarity(embed_i, embed_j)
                
                # Always ensure the patent IDs are ordered consistently
                # This ensures we always compute sim(A,B) and never sim(B,A)
                if id_i < id_j:
                    similarity_results.append((id_i, id_j, sim))
                else:
                    similarity_results.append((id_j, id_i, sim))
                
                results_count += 1
                processed_pairs_count += 1
                
                # Write to disk if threshold reached
                if results_count >= SAVE_THRESHOLD:
                    output_file = os.path.join(OUTPUT_DIR, f"similarities_rank{rank}_part{file_counter}.parquet")
                    df = pd.DataFrame(similarity_results, columns=['patent_id_1', 'patent_id_2', 'similarity'])
                    
                    table = pa.Table.from_pandas(df, schema=schema)
                    pq.write_table(table, output_file)
                    
                    print(f"Rank {rank}: Saved {results_count} similarities to {output_file}")
                    print(f"Rank {rank}: Total processed pairs so far: {processed_pairs_count}")
                    
                    # Free memory explicitly
                    del similarity_results
                    del df
                    del table
                    gc.collect()
                    
                    similarity_results = []
                    results_count = 0
                    file_counter += 1
        
        # Clean up memory after each block pair
        del block_i_embed
        if block_i != block_j:
            del block_j_embed
        gc.collect()
    
    # Save any remaining results
    if similarity_results:
        output_file = os.path.join(OUTPUT_DIR, f"similarities_rank{rank}_part{file_counter}.parquet")
        df = pd.DataFrame(similarity_results, columns=['patent_id_1', 'patent_id_2', 'similarity'])
        
        table = pa.Table.from_pandas(df, schema=schema)
        pq.write_table(table, output_file)
        print(f"Rank {rank}: Saved final {len(similarity_results)} similarities to {output_file}")
    
    # Final checkpoint save
    
    total_time = time.time() - start_time
    print(f"Rank {rank}: Processing completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Rank {rank}: Total pairs processed: {processed_pairs_count}")
    
    # Wait for all processes to finish
    comm.Barrier()
    
    # Final summary from rank 0
    if rank == 0:
        print("All processes completed.")
        print("If job was interrupted, you can resume by:")
        print("1. Running your parsing script on the new .out file")
        print("2. Updating RESUME_PROGRESS vector in this script")
        print("3. Setting RESUME_FROM_CHECKPOINT = True")

if __name__ == "__main__":
    main()