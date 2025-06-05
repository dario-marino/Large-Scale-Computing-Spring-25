# Final Project - Dario Marino
## Of Rivalry and Sinergy: The Role of Patents and Products Similarity on R&D Investment

You can see the slides for my Master Thesis [here](https://github.com/macs30123-s25/final-project-dariello/blob/main/Slides_MACSS_Thesis-1.pdf).

My Economics thesis investigates the effects of patent similarity on firms' R&D investments, distinguishing between within-sector and between-sector technological spillovers. Building on a Cournot competition framework, the analysis posits that within-sector patent similarity inhibits R&D activity, as firms are reluctant to invest in innovations that may benefit competitors. In contrast, between-sector patent similarity is expected to stimulate R&D by facilitating knowledge transfer across industries without competitive risks.

To empirically test these hypotheses, a novel dataset is developed by integrating USPTO patent claims (1976–2014) with firm-level financial data from Compustat US. Patent similarity metrics are computed using advanced natural language processing methods, including Doc2Vec and cosine similarity scores, which represent the semantic relationships between patent claims.

The research employs Generalized Method of Moments (GMM) estimation to assess the impact of these variables on firm-level R&D investments. Controls include firm patent portfolio size, market concentration (Herfindahl–Hirschman Index), and patent novelty, which are expected to modulate the observed relationships. Additional robustness checks incorporate recent advancements in patent similarity modeling using BERT-based representations, as well as comparisons with global datasets like PATSTAT.

Preliminary findings from simulations with 1000 firms suggest a nuanced role for patent similarity: while within-sector similarity negatively correlates with R&D, between-sector similarity shows a positive effect, with the strongest impacts observed in cross-sectoral spillovers. These results have significant policy implications, advocating for differentiated patent protection regimes that reduce barriers to inter-sector knowledge diffusion while safeguarding intra-sector innovation from competitive spillovers. The study contributes to the literature on innovation economics by advancing methods for quantifying technological similarity and exploring its role in shaping firms' innovation strategies. All data and code will be shared openly on Zenodo to facilitate replication and further research.

To estimate similarity I downloaded the abstract of all patents granted in the US from 1976 to 2024 from the PatentsView database: [g_patent_abstract](https://patentsview.org/download/data-download-tables). I then used Modern Bert to assign word embeddings for each patent. I observed that the distribution of words was 249 at the 99th percentile, which is within my max length of tokens of 512 (512*0.75=384). My parallelized code that I am presenting for the final project computes the pairwise similarities of 8 million patents, this means that we should perform 64 trillions cosine similarity operations, and each word embedding is around 300 columns. I already run my code for 30 hours, I used all 128 AMD cores. I didn't finish to compute all the similarities, because it would take more than my available computing units to do so. I talked about this with Jon and we agreed on the fact that my operation showed that my code works and it is scalable, but it's just handling a problem which is too large for the computing units available. I will show the amount of data that I have (We are in the order of billions patent similarities computed) and the output of the code. I need this data for my thesis so I will try to access the Midway SSD resources to compute these similarities during the summer.

Before starting the analysis I had to install certain packages using this [.sh](https://github.com/macs30123-s25/final-project-dariello/blob/main/macs30123_install.sh). The main code for the pairwise similarity computation that I have called on AMD for 128 cores can be found [here](https://github.com/macs30123-s25/final-project-dariello/blob/main/optimized_pair.py), and this is the [sbatch](https://github.com/macs30123-s25/final-project-dariello/blob/main/optimized.sbatch) file that I have used to call it. After we have the folder with all the parquet parts we use this [code](https://github.com/macs30123-s25/final-project-dariello/blob/main/combine.py) to combine the files in one unique pairwise similarity parquet dataset and I use the following [sbatch](https://github.com/macs30123-s25/final-project-dariello/blob/main/combine.sbatch) file to call it. I am going through different part of the [main code](https://github.com/macs30123-s25/final-project-dariello/blob/main/optimized_pair.py) in the next output code points.

## Output Code Points

### 0. Parallel Bert (32 Cores)

I used [ModernBert](https://huggingface.co/blog/modernbert) which is a recent improvement (Christmas 2024) for Bert which could process 8k tokens instead of the usual 512. For this task Modern Bert is actually overkilling it because we don't need more than 512 tokens for the reson that I explained before. I used this code on Acropolis, with a graphical interface. I called Spyder using qsub with 200 GB of RAM and 32 cores and I parallelized this process using this [code](https://github.com/macs30123-s25/final-project-dariello/blob/main/UMBERTOD.py). This is the qsub that I have used to call spyder:

```
qsub -N icewm-spyder -l nodes=1:ppn=32,mem=250g -I -X -x /share/overflow/spyder.sh -M dariomarino@uchicago.edu -m a
```


### 1. Distributed Computing with MPI

The implementation leverages MPI for parallel processing across multiple nodes or cores. This distributed computing approach:

- Divides the similarity computation workload evenly across available processors
- Minimizes inter-process communication by assigning independent block pairs to each process
- Uses barrier synchronization to coordinate process completion
- Scales efficiently with additional computational resources

```
# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```



### 2. Memory-Efficient Data Handling

The code implements several sophisticated memory management techniques:

- **Block Processing**: Data is processed in configurable blocks (`BLOCK_SIZE`), allowing the system to handle datasets much larger than available RAM. This chunking strategy enables the processing of millions of patents with our limited memory of 256 GB of RAM for each amd core as can be seen here in the [partitions configurations](https://rcc-uchicago.github.io/user-guide/partitions/#configurations). Results are also periodically saved to disk based on a configurable threshold (`SAVE_THRESHOLD`), preventing the accumulation of too many results in memory at once.

```
# Configuration
EMBEDDING_FILE = "/home/dariomarino/Downloads/patent_embeddings.npy"
PATENT_IDS_FILE = "/home/dariomarino/Downloads/patent_ids.csv"
OUTPUT_DIR = "/scratch/midway3/dariomarino/similarity_output"
BLOCK_SIZE = 500                                 
SAVE_THRESHOLD = 1_000_000     
```


- **Memory Mapping**: Instead of loading the entire embedding matrix into RAM, which would make the code fail almost immediately, the code uses NumPy's memory mapping (`mmap_mode='r'`) to access the embedding file directly from disk. Only the required portions are loaded into memory when accessed, significantly reducing the memory footprint. This will make the code slower compared to accessing data from RAM but there was no other faster way to access data. As we saw in the slides accessing data from disk is the  best alternative compared to RAM.

![image](https://github.com/macs30123-s25/final-project-dariello/blob/main/memory.png)

```
      # Memory-map the embedding file instead of loading it all at once
    embeddings = np.load(EMBEDDING_FILE, mmap_mode='r')
    
    if rank == 0:
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Using block size: {BLOCK_SIZE}")
    
    # Calculate the total number of blocks
    num_blocks = math.ceil(num_patents / BLOCK_SIZE)
```


- **Explicit Memory Management**: The implementation actively manages memory by explicitly deleting temporary objects and forcing garbage collection after each batch is processed and saved. This prevents memory leakage during long-running computations.

```
        # Clean up memory after each block pair
        del block_i_embed
        if block_i != block_j:
            del block_j_embed
        gc.collect()
```


### 3. Computational Efficiency Optimizations

Several algorithmic optimizations reduce unnecessary computations:

- **Upper Triangular Matrix Computation**: The code only computes the upper triangular portion of the similarity matrix (including diagonal), exploiting the symmetrical nature of cosine similarity (sim(A,B) = sim(B,A)). This reduces our computational workload (and time spent) by 50%, going from 64 trillions of patent similarities to 32 trillions of patent similarities.

```
    # Distribute block pairs among processes
    block_pairs = []
    pair_idx = 0
    
    for i in range(num_blocks):
        for j in range(i, num_blocks):  # Only upper triangle including diagonal
            if pair_idx % size == rank:
                block_pairs.append((i, j))
            pair_idx += 1
    
    if rank == 0:
        print(f"Each process will handle approximately {total_block_pairs // size} block pairs")
```


- **Block-Pair Distribution**: Work is distributed as block pairs rather than individual comparisons, reducing scheduling overhead while maintaining load balance across processes.

```
    for block_idx, (block_i, block_j) in enumerate(block_pairs):
        if block_idx % 10 == 0:
            print(f"Rank {rank}: Processing block pair {block_idx + 1}/{len(block_pairs)} ({block_i},{block_j})")
        
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
```


- **Consistent ID Ordering**: Patent pairs are always stored with the smaller ID first, ensuring consistency and avoiding duplicate calculations of the same pair in different orders.

```
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
```


### 4. Storage

The implementation uses robust data handling approaches:

- **PyArrow and Parquet**: Results are saved in the columnar Parquet format using PyArrow, providing excellent compression and query performance for downstream analysis. Results are organized into multiple Parquet files with a consistent schema, facilitating easy aggregation and analysis after computation.

```
if results_count >= SAVE_THRESHOLD:
                    output_file = os.path.join(OUTPUT_DIR, f"similarities_rank{rank}_part{file_counter}.parquet")
                    df = pd.DataFrame(similarity_results, columns=['patent_id_1', 'patent_id_2', 'similarity'])
                    
                    table = pa.Table.from_pandas(df, schema=schema)
                    pq.write_table(table, output_file)
```



### 5. Dependencies

- NumPy: For efficient numerical operations
- pandas: For data manipulation and management
- PyArrow: For Parquet file handling
- mpi4py: Python interface for MPI parallel processing
- SciPy: For cosine distance calculation


### 6. Algorithm Workflow

1. Initialize MPI environment and distribute work among processes
2. Memory-map the embedding file for efficient access
3. Divide patents into blocks and distribute block pairs among processes
4. For each block pair:
   - Load relevant embedding blocks into memory
   - Compute similarities between patents in the blocks
   - Apply optimizations to avoid redundant calculations
   - Store results with consistent patent ID ordering
5. Periodically save accumulated results to Parquet files
6. Clean up memory after each block pair and saved batch
7. Synchronize processes upon completion


This is how the process looks like for each core:

```
Rank 66: Saved 1000000 similarities to /scratch/midway3/dariomarino/similarity_output/similarities_rank66_part1619.parquet
Rank 66: Saved 1000000 similarities to /scratch/midway3/dariomarino/similarity_output/similarities_rank66_part1620.parquet
Rank 66: Saved 1000000 similarities to /scratch/midway3/dariomarino/similarity_output/similarities_rank66_part1621.parquet
Rank 66: Processing block pair 6491/1287019
Rank 66: Saved 1000000 similarities to /scratch/midway3/dariomarino/similarity_output/similarities_rank66_part1622.parquet
Rank 66: Saved 1000000 similarities to /scratch/midway3/dariomarino/similarity_output/similarities_rank66_part1623.parquet
Rank 66: Processing block pair 6501/1287019
Rank 66: Saved 1000000 similarities to /scratch/midway3/dariomarino/similarity_output/similarities_rank66_part1624.parquet
Rank 66: Saved 1000000 similarities to /scratch/midway3/dariomarino/similarity_output/similarities_rank66_part1625.parquet
Rank 66: Saved 1000000 similarities to /scratch/midway3/dariomarino/similarity_output/similarities_rank66_part1626.parquet
Rank 66: Processing block pair 6511/1287019
```


I am sure that there are no duplicates of operations thanks to rank-based assignment. The critical line is (`pair_idx % size == rank`) which ensures each block pair is assigned to exactly one rank based on the modulo operation. 
With 128 ranks:

Rank 0 gets indices: 0, 128, 256, ...
Rank 1 gets indices: 1, 129, 257, ...
Rank 2 gets indices: 2, 130, 258, ...
Rank 3 gets indices: 3, 131, 259, ...
Rank 4 gets indices: 4, 132, 260, ...

### 7. Output Format

The output consists of multiple Parquet files containing:
- `patent_id_1`: First patent identifier (always the lexicographically smaller ID)
- `patent_id_2`: Second patent identifier
- `similarity`: Cosine similarity score between the patents (range 0-1)

These files can be easily combined using [combine.py](https://github.com/macs30123-s25/final-project-dariello/blob/main/combine.py). My data are already available on scratch, where will remain available for one month. On (`/scratch/midway3/dariomarino`) you can see a folder called *similarity_output1* with the raw parquet files and the file *similarities.parquet* which is the combined parquet files and the dataset that I will use for my future technology similarity analysis. On (`/home/dariomarino/Downloads
`) you can see the sbatch and python files that are also available in this repository, and you can also see the patent embeddings and the patent_id csv files that are used as input for the main code. On (`/home/dariomarino`) you can see the .out and .err files that are given after the codes have finished running.

I also uploaded one [parquet similarity file](https://github.com/macs30123-s25/final-project-dariello/blob/main/similarities_rank0_part0.parquet) here for you to see, it is just the two patent ids and the similarity between them.

You can see here what I have after I ran it for just 30 hours and then I ran the combine code on one core on caslake. I was able to have 900 GB of parquet data file, and I still have a lot to process but the process is definitely scalable in a consistent way:

![allafineditutto](https://github.com/macs30123-s25/final-project-dariello/blob/main/allafineditutto.png)


I was also able to use dask cuda and cudf for this process. I think I still feel better using the cpu solution but I could work on the GPU solution too if I think it could be helpful. Here is the [code](https://github.com/macs30123-s25/a3-generatoredi/blob/main/gpu.py) and here is an output example


```
Saved 16000000 similarities to /scratch/midway3/dariomarino/similarity_output_gpu/similarities_part3130.parquet
Processing batch 3132/2574748 with 4 block pairs
Saved 16000000 similarities to /scratch/midway3/dariomarino/similarity_output_gpu/similarities_part3131.parquet
Processing batch 3133/2574748 with 4 block pairs
Saved 16000000 similarities to /scratch/midway3/dariomarino/similarity_output_gpu/similarities_part3132.parquet
Processing batch 3134/2574748 with 4 block pairs
Saved 16000000 similarities to /scratch/midway3/dariomarino/similarity_output_gpu/similarities_part3133.parquet
Processing batch 3135/2574748 with 4 block pairs
Saved 16000000 similarities to /scratch/midway3/dariomarino/similarity_output_gpu/similarities_part3134.parquet
Processing batch 3136/2574748 with 4 block pairs
Saved 16000000 similarities to /scratch/midway3/dariomarino/similarity_output_gpu/similarities_part3135.parquet
Processing batch 3137/2574748 with 4 block pairs
Saved 16000000 similarities to /scratch/midway3/dariomarino/similarity_output_gpu/similarities_part3136.parquet
Processing batch 3138/2574748 with 4 block pairs
Saved 16000000 similarities to /scratch/midway3/dariomarino/similarity_output_gpu/similarities_part3137.parquet
Processing batch 3139/2574748 with 4 block pairs
Saved 16000000 similarities to /scratch/midway3/dariomarino/similarity_output_gpu/similarities_part3138.parquet
Processing batch 3140/2574748 with 4 block pairs
[dariomarino@midway3-login3 ~]$
```

![gpu](https://github.com/macs30123-s25/final-project-dariello/blob/main/gpufinal.png)



After I hit the time limit I can just check the out file using this [code](https://github.com/macs30123-s25/final-project-dariello/blob/main/where.py) which will return me a vector of the progress made in each rank that I can put in the [vectorized original code](https://github.com/macs30123-s25/final-project-dariello/blob/main/vectorized_pair.py) so that each worker can start back from where I was interrupted for the next 30 hours.


![imagevector](https://github.com/macs30123-s25/final-project-dariello/blob/main/vectorized.png)
