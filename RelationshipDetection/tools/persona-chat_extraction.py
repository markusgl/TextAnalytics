"""
Extracts dialogs from Persona-Chat corpora
"""

import spacy
import re

from spacy.symbols import PERSON
from nltk.tokenize import sent_tokenize


nlp = spacy.load('en')

relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'mommy', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'grampa', 'grandpa', 'grandma', 'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'brother-in-law',
                     'sister-in-law', 'husband', 'wife']
me_list =['i', 'me', 'my']


def write_file(text):
    # out file
    with open('../data/validation/persona-chat_conversations.txt', 'a') as f:
        f.write(text)


def search_entity_or_relation(sentence, extract_rels=False):
    """
    checks if at least two persons or on relations appears within a sentence
    :param sentence:
    :param extract_rels:
    :return:
    """
    count_person = 0
    count_me = 0
    doc = nlp(sentence)

    for ent in doc.ents:
        if ent.label == PERSON:
            count_person += 1

    if count_person >= 2:
        write_file(sentence + '\n')

    elif extract_rels:
        for token in doc:
            if token.text.lower() in relationship_list:
                write_file(sentence + '\n')
    else:
        for token in doc:
            if token.text.lower() in relationship_list:
                count_me += 1

    if count_me > 0 and count_person > 0:
        write_file(sentence + '\n')


# in file
with open('../data/ConvAI2/train_none_original_no_cands.txt', 'r') as f:
    count = 0
    min_val = 0
    max_val = 100000

    for line in f.readlines():
        line = re.sub('\s{2,}', ' ', line)
        line = line[2:]
        line = line.replace('__SILENCE__', '')
        if len(line) > 3:
            write_file(line)

        """
        
        if count == max_val:
            break
        elif count >= min_val:
            #line = re.sub('\W+', ' ', line)
            line = re.sub('\s{2,}', ' ', line)
            line = line[2:]
            line = line.replace('__SILENCE__', '')
            if len(line) > 3:
                write_file(line)
            #for sentence in sent_tokenize(line):
            #    search_entity_or_relation(sentence, extract_rels=False)

        count += 1
        """


