import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# getting data
url = 'https://github.com/BobAdamsEE/SouthParkData/blob/master/All-seasons.csv?raw=true'
df = pd.read_csv(url)

# get rid of new lines and fillers
df["Line"] = df["Line"].str.replace('\n', '', regex=True)
df["Line"] = df["Line"].str.replace('uh', '', regex=True)
df["Line"] = df["Line"].str.replace('um', '', regex=True)

# get lines as list
lines = df["Line"].values.tolist()

# get model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('Xuhui/ToxDect-roberta-large')
model = AutoModelForSequenceClassification.from_pretrained('Xuhui/ToxDect-roberta-large')

# Process in chunks of 1000
batch_size = 1000
dfs = []  # List to store partial results

# tokenize all at once and restrict max length
inputs = tokenizer(lines, return_tensors="pt", padding=True, truncation=True)
tokenized_lines = [tokenizer.convert_ids_to_tokens(input) for input in inputs["input_ids"]]
df["token_lines"] = tokenized_lines
df["token_line_len"] = [len(line) for line in tokenized_lines]

# find lines less than 115 (99.5% percentile) tokens
df_reduced = df[df["token_line_len"] < 115].copy()
lines_reduced = df_reduced["Line"].tolist()

for i in range(0, len(lines_reduced), batch_size):
    batch_lines = lines_reduced[i:i + batch_size]  # Get batch
    batch_inputs = tokenizer(batch_lines, return_tensors="pt", padding=True, truncation=True)

    # Run model
    logits = model(batch_inputs["input_ids"], batch_inputs["attention_mask"]).logits
    probs = 100 * torch.softmax(logits, dim=1)[:, 1].detach().numpy()
    
    # Convert to list of floats
    float_probs = [float(prob) for prob in probs]
    
    # Store results in a temporary dataframe
    df_batch = df_reduced.iloc[i:i + batch_size].copy()
    df_batch["prob"] = float_probs
    dfs.append(df_batch)
    
# Concatenate all batch dataframes into final dataframe
df_final = pd.concat(dfs, ignore_index=True)

# Save to CSV
df_final.to_csv("sp_toxigen_v2.csv", index=False)
