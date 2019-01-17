import pandas as pd

df = pd.read_csv('../RelationshipDetection/data/simpsons-by-the-data/simpsons_script_lines.csv', error_bad_lines=False)

spoken_words = df['raw_text']

with open('../RelationshipDetection/data/simpons_conversations_character.txt', 'w', encoding='utf-8') as f:
    for row in spoken_words:
        if not str(row) == 'nan':
            f.write(str(row) + '\n')


