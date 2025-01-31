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
probs = []

# get model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('Xuhui/ToxDect-roberta-large')
model = AutoModelForSequenceClassification.from_pretrained('Xuhui/ToxDect-roberta-large')#.eval()

for line in df["Line"]:
    input = tokenizer(line, return_tensors="pt")["input_ids"]
    #if len(input["input_ids"]) < 115:
    logit = model(input).logits
    prob = 100 * float(torch.softmax(logit, dim=1)[:, 1].detach().numpy())
    probs.append(prob)
    break

# combine probabilities back with season/ep/char data, save
df_lines = df.copy()
df_lines["prob"] = probs
df_lines.to_csv("sp_toxigen_v5.csv")