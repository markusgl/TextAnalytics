"""
Extracts dialogs from Persona-Chat corpora
"""

import spacy
import re

from spacy.symbols import PERSON
from nltk.tokenize import sent_tokenize

from random import randint

nlp = spacy.load('en')

relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'mommy', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'grampa', 'grandpa', 'grandma', 'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'brother-in-law',
                     'sister-in-law', 'husband', 'wife']


def search_entity_or_relation(sentence, extract_rels=False):
    doc = nlp(sentence)

    for ent in doc.ents:
        if ent.label == PERSON:
            return True

    if extract_rels:
        for token in doc:
            if token.text.lower() in relationship_list:
                return True

    return False


# in file
with open('../data/ConvAI2/train_none_original_no_cands.txt', 'r') as f:
    corpus = ''
    data = f.readlines()
    count = 0
    min_val = 0
    max_val = 5000

    for line in data:
        if count == max_val:
            break
        elif count >= min_val:
            #line = re.sub('\W+', ' ', line)
            line = re.sub('\s{2,}', ' ', line)
            line = line[2:]
            line = line.replace('__SILENCE__', '')
            for sentence in sent_tokenize(line):
                if search_entity_or_relation(sentence, extract_rels=True):
                    corpus += sentence + '\n'

        count += 1

# out file
with open('../data/ConvAI2/extracted_conversations_with_persons-rels.txt', 'a') as f:
    f.write(corpus)

