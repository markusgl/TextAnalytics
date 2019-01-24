import spacy
import logging
import re

from spacy.symbols import PERSON
from nltk.tokenize import sent_tokenize

nlp = spacy.load('en')

relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'mommy', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'grampa', 'grandpa', 'grandma', 'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'brother-in-law',
                     'sister-in-law', 'husband', 'wife']


def search_entity_or_relation(sentence):
    sentence = re.sub('\s{2,}', ' ', sentence)
    doc = nlp(sentence)
    extract = False

    for ent in doc.ents:
        if ent.label == PERSON:
            print(ent.text)
            extract = True

    #for token in doc:
    #    if token.text.lower() in relationship_list:
    #        extract = True

    return extract


with open('../data/ConvAI2/extracted_conversations.txt', 'r') as f:
    corpus = ''
    data = f.readlines()
    count = 0
    min_val = 15000
    max_val = 20000

    for line in data:
        if count == max_val:
            break
        elif count >= min_val:
            if search_entity_or_relation(line):
                corpus += line

        count += 1

# out file
with open('../data/ConvAI2/extracted_conversations_with_persons.txt', 'a') as f:
    f.write(corpus)

