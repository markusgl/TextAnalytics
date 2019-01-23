import json
import re
import pandas as pd
import spacy

from flair.data import Sentence
from flair.models import SequenceTagger
from nltk.tokenize import sent_tokenize, word_tokenize
from pprint import pprint

relationship_list_de = ['vater', 'mutter', 'papa', 'papi', 'mama', 'mami', 'sohn', 'tochter', 'bruder', 'schwester',
                     'enkel', 'enkelin', 'nichte', 'neffe', 'großvater', 'großmutter', 'opa', 'opa',
                     'onkel', 'tante', 'cousin', 'cousine', 'schwager', 'schwägerin', 'mann', 'frau', 'ehemann',
                     'ehefrau']

relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'mommy', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'grampa', 'grandpa', 'grandma', 'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'brother-in-law',
                     'sister-in-law', 'husband', 'wife']

nlp = spacy.load('en')


def extract_human_conversations(df, persist=False):
    # save all human conversations as continuous text
    human_conv = df.loc[df['sender'] == 'Human']

    corpus = ""
    for row in human_conv['text']:
        #corpus += row.replace('\n', '') + ' '
        corpus += row + '\n'

    if persist:
        with open('../data/human_dialog.txt', 'w') as f:
            f.write(corpus)

    return corpus


def generate_dataframe_from_json(data):
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


def extract_dialog_data(data):
    corp = ''
    for row in data:
        for dialog in row['dialog']:
            text = dialog['text']

            # remove unicode characters
            text = re.sub(r'[^\x00-\x7F]', ' ', text)
            if text:
                corp += text + '\n'

    return corp


def extract_dialogs_from_training_data(json_data):
    corp = ''
    for row in json_data:
        for dialog in row['dialog']:
            text = dialog['text']

            # remove unicode characters
            text = re.sub(r'[^\x00-\x7F]', '', text)
            if text:
                corp += text + '\n'

    with open('../data/convai/export_dialogs.txt', 'a') as f:
        f.write(corp)


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


def named_entity_tagger_spacy(corpus):
    entity_sentences = []

    for sentence in sent_tokenize(corpus):
        doc = nlp(sentence)

        # check if either a person or a social relation appear within the sentence
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entity_sentences.append(sentence)
                break

        for token in doc:
            if token.text.lower() in relationship_list:
                entity_sentences.append(sentence)
                break

    return entity_sentences


def named_entity_tagger_flair(corpus):
    entity_sentences = []

    #for line in sent_tokenize(corpus):
    for line in corpus:
        #clean_sentence = re.sub('\W+', ' ', line)  # remove non-word characters
        sentence = Sentence(line)
        tagger = SequenceTagger.load('ner')
        tagger.predict(sentence)

        for entity in sentence.get_spans('ner'):
            if entity.tag == 'PER':
                entity_sentences.append(line)

        for token in word_tokenize(line):
            if token.lower() in relationship_list:
                entity_sentences.append(line)

    return entity_sentences


def write_to_file(out_file, data):
    with open(out_file, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(item + '\n')


#data = load_json('../data/convai/data_tolokers.json')
#dataframe = generate_dataframe_from_json(data)
#corp = extract_human_conversations(dataframe)
#ent_sentences = named_entity_tagger_spacy(corp)


data = load_json('../data/convai/data_volunteers.json')
corp = extract_dialog_data(data)
ent_sentences = named_entity_tagger_spacy(corp)
write_to_file('../data/convai/human-bot-conversations-with-relations.txt', ent_sentences)

