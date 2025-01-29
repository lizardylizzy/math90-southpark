import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Create output directory if it doesn't exist
output_dir = "toxigenv4_outputs"
os.makedirs(output_dir, exist_ok=True)

# Parse batch number from command line arguments
parser = argparse.ArgumentParser(description="Process a specific batch of lines")
parser.add_argument("batch_number", type=int, help="Batch number to process (starting from 1)")
args = parser.parse_args()

# Define batch size
batch_size = 1000

# Load data
url = 'https://github.com/BobAdamsEE/SouthParkData/blob/master/All-seasons.csv?raw=true'
df = pd.read_csv(url)

# Clean text
df["Line"] = df["Line"].str.replace('\n', '', regex=True)
df["Line"] = df["Line"].str.replace('uh', '', regex=True)
df["Line"] = df["Line"].str.replace('um', '', regex=True)

# Get total batches
total_batches = (len(df) + batch_size - 1) // batch_size

# Ensure batch_number is valid
if args.batch_number < 1 or args.batch_number > total_batches:
    print(f"Error: Batch number {args.batch_number} is out of range (1-{total_batches})")
    exit(1)

# Get start and end indices for the requested batch
start_idx = (args.batch_number - 1) * batch_size
end_idx = min(start_idx + batch_size, len(df))

# Extract batch lines
batch_lines = df["Line"].iloc[start_idx:end_idx].tolist()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('Xuhui/ToxDect-roberta-large')
model = AutoModelForSequenceClassification.from_pretrained('Xuhui/ToxDect-roberta-large')

# Tokenize batch
inputs = tokenizer(batch_lines, return_tensors="pt", padding=True, truncation=True)

# Run model
logits = model(inputs["input_ids"], inputs["attention_mask"]).logits
probs = 100 * torch.softmax(logits, dim=1)[:, 1].detach().numpy()

# Convert to float
float_probs = [float(prob) for prob in probs]

# Store results in dataframe
df_batch = df.iloc[start_idx:end_idx].copy()
df_batch["prob"] = float_probs

# Save batch to CSV
batch_filename = os.path.join(output_dir, f"batch_{args.batch_number}.csv")
df_batch.to_csv(batch_filename, index=False)

print(f"Batch {args.batch_number} processed and saved to {batch_filename}")