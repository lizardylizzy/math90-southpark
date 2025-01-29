import pandas as pd
from googleapiclient import discovery

API_KEY = 'AIzaSyDzBhiGi_uBXqDPqPUdjNaSbmDmzATAwaU'

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
scores = []

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

for line in lines:
    
    analyze_request = {
    'comment': { 'text': line},
    'requestedAttributes': {'TOXICITY': {}} # 'IDENTITY_ATTACK': {}, 'INSULT': {}, 'PROFANITY': {}, 'THREAT': {}
    }
    try:
        response = client.comments().analyze(body=analyze_request).execute()
        score = response.get('attributeScores', {}).get('TOXICITY', {}).get('summaryScore', {}).get('value', None)
        if score is not None:
            scores.append(score)
    except Exception as e:
        scores.append(None)

# save scores
df_lines = df.copy()
df_lines["prob"] = scores
df_lines.to_csv("sp_perspective.csv")