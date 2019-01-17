import json
import re
import pandas as pd

from pprint import pprint

with open('../data/data_intermediate.json', 'r') as f:
    data = json.load(f)

#print(type(data))
#pprint(data)

df_columns = ['sender', 'text']
df = pd.DataFrame(columns=df_columns)

for row in data:
    for dialog in row['dialog']:
        sender = dialog['sender_class']
        text = dialog['text']

        # remove unicode characters
        text = re.sub(r'[^\x00-\x7F]', ' ', text)

        data = {'sender': sender, 'text': text}
        df = df.append(data, ignore_index=True)

print(df.head())

# save all human conversations as continuous text
human_conv = df.loc[df['sender'] == 'Human']

corpus = ""
for row in human_conv['text']:
    #print(row)
    corpus += row.replace('\n', '') + ' '

print(corpus)

with open('../data/human_dialog.txt', 'w') as f:
    f.write(corpus)
