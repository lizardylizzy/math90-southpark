import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# getting data
url = 'https://github.com/BobAdamsEE/SouthParkData/blob/master/All-seasons.csv?raw=true'
df = pd.read_csv(url)
df = df[:100]

# get rid of new lines
df["Line"] = df["Line"].str.replace('\n', '', regex=True)

# replace fillers
df["Line"] = df["Line"].str.replace('uh', '', regex=True)
df["Line"] = df["Line"].str.replace('um', '', regex=True)

# get lines as list
lines = df["Line"].values.tolist()

# get model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('Xuhui/ToxDect-roberta-large')
model = AutoModelForSequenceClassification.from_pretrained('Xuhui/ToxDect-roberta-large')#.eval()

# tokenize lines and store tokenized version and length
inputs = tokenizer(lines, return_tensors="pt", padding=True, add_special_tokens = True) # can this take a list
tokenized_lines = [tokenizer.convert_ids_to_tokens(input) for input in inputs["input_ids"]]
df["token_lines"] = tokenized_lines
df["token_line_len"] = [len(line) for line in tokenized_lines]

# find lines less than 115 (99.5% percentile) tokens
indices_to_keep = df.index[df["token_line_len"] < 100].tolist()
print(indices_to_keep)
indices_tensor = torch.tensor(indices_to_keep, dtype=torch.long)
inputs_shortened = {
    "input_ids": inputs["input_ids"][indices_tensor],
    "attention_mask": inputs["attention_mask"][indices_tensor]
}
print(inputs_shortened)

logits = model(inputs["input_ids"], inputs["attention_mask"]).logits
probs = 100 * torch.softmax(logits, dim=1)[:, 1].detach().numpy()
float_probs = [float(prob) for prob in probs]

print(float_probs)
# combine probabilities back with season/ep/char data, save
df_lines = df.loc[indices_to_keep].copy()
df_lines["prob"] = float_probs
df_lines.to_csv("sp_toxigen.csv")