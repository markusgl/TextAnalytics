import pandas as pd
import spacy
import en_core_web_sm

from spacy.symbols import PERSON
from nltk.tokenize import sent_tokenize

nlp = en_core_web_sm.load()
relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'husband', 'wife']

me_list = ['i', 'me', 'my']


def export_conversations():
    data = pd.read_csv('../data/Friends_TV_Corpus/friends-final.txt', sep='\t')
    conversations = data['line']
    conversations.to_csv('../data/Friends_TV_Corpus/friends_conversations.txt', header=None, index=None, sep=' ', mode='a')


def tag_person_entities(line):  # PER-PER  ME-PER
    for sentence in sent_tokenize(line):
        count_persons = 0
        count_me = 0
        doc = nlp(sentence)

        for ent in doc.ents:
            if ent.label == PERSON:
                count_persons += 1

        if count_persons > 1:
            return True
        else:
            for token in sent_tokenize(sentence):
                if token in me_list:
                    count_me += 1

        if count_me > 0 and count_persons > 0:
            return True

        return False


person_convs = ''
with open('../data/simpsons_conversations.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if tag_person_entities(line):
            line = line.replace('"', '')
            person_convs += line

with open('../data/validation/training_set/simpsons_training_per-per_me-per.txt', 'a', encoding='utf-8') as f:
    f.write(person_convs)
