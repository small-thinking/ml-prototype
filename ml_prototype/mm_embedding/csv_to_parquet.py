import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm  # For progress bar
import os

# Paths
csv_file_path = os.path.expanduser("~/Downloads/characters.csv")
parquet_file_path = os.path.expanduser("~/Downloads/characters.snappy.parquet")

# Define chunk size (number of rows per chunk)
chunk_size = 1000

# Initialize progress tracking
total_rows = sum(1 for _ in open(csv_file_path)) - 1  # Exclude header row
progress = tqdm(total=total_rows, desc="Processing rows", unit="rows")

# Create a list to hold chunks temporarily
chunks = []

# Process CSV in chunks
for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
    chunks.append(pa.Table.from_pandas(chunk))
    progress.update(chunk.shape[0])

# Close progress bar
progress.close()

# Combine all chunks into a single Parquet table
full_table = pa.concat_tables(chunks)

# Write the full table to a Snappy-compressed Parquet file
pq.write_table(full_table, parquet_file_path, compression="snappy")

print(f"Parquet file saved at {parquet_file_path}")