import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

#First R code to see where we can push embeddings

# Load TSV
df = pd.read_csv('/home/dariomarino/g_patent_abstract.tsv', sep='\t')

# Load Modern BERT model
model_name = "bert-base-uncased"  # Replace with Modern BERT path if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Function to get mean pooled embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512) #word abstract
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to process a batch of abstracts
def process_batch(start, end, abstracts):
    embeddings = []
    for abstract in abstracts[start:end]:
        embeddings.append(get_embedding(str(abstract)))
    return embeddings

# Number of processes (based on cores)
num_processes = 31  

# Split the dataset into chunks for parallel processing
chunk_size = len(df) // num_processes
chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
if len(df) % num_processes != 0:
    chunks[-1] = (chunks[-1][0], len(df))  # Handle remainder in the last chunk

# Use ProcessPoolExecutor to parallelize the process
embeddings = []
with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = []
    for start, end in chunks:
        futures.append(executor.submit(process_batch, start, end, df['patent_abstract'].tolist()))
    
    # Collect results
    for future in tqdm(as_completed(futures), total=len(futures)):
        embeddings.extend(future.result())

# Save embeddings and patent IDs
np.save('/home/dariomarino/patent_embeddings.npy', np.array(embeddings))
df[['patent_id']].to_csv('/home/dariomarino/patent_ids.csv', index=False)
