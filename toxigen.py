import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# getting data
url = 'https://github.com/BobAdamsEE/SouthParkData/blob/master/All-seasons.csv?raw=true'
df = pd.read_csv(url)

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

# Test tokenizer on a random line
# print('Original: ', lines[20003])
# print('Tokenized: ', tokenizer.tokenize(lines[20003]))
# print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lines[20003])))

# run model on first 50 lines
inputs = tokenizer(lines, return_tensors="pt", padding=True, add_special_tokens = True) # can this take a list
logits = model(inputs["input_ids"], inputs["attention_mask"]).logits
probs = 100 * torch.softmax(logits, dim=1)[:, 1].detach().numpy()
float_probs = [float(prob) for prob in probs]

# combine probabilities back with season/ep/char data, save
df_lines = df.copy()
df_lines["prob"] = float_probs
df_lines.to_csv("sp_toxigen.csv")