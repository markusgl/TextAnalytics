import json
import re
import pandas as pd

from pprint import pprint


def extract_human_conversations(df, persist=False):
    # save all human conversations as continuous text
    human_conv = df.loc[df['sender'] == 'Human']

    corpus = ""
    for row in human_conv['text']:
        # print(row)
        corpus += row.replace('\n', '') + ' '

    if persist:
        with open('../data/human_dialog.txt', 'w') as f:
            f.write(corpus)

    return corpus


def dataframe_from_json(data):
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

    return df


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


load_json('../data/convai/data_intermediate.json')




