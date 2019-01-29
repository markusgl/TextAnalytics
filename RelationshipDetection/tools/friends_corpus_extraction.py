import pandas as pd
import spacy
import en_core_web_sm

from spacy.symbols import PERSON
from nltk.tokenize import sent_tokenize

nlp = en_core_web_sm.load()
relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'husband', 'wife']


def export_conversations():
    data = pd.read_csv('../data/Friends_TV_Corpus/friends-final.txt', sep='\t')
    conversations = data['line']
    conversations.to_csv('../data/Friends_TV_Corpus/friends_conversations.txt', header=None, index=None, sep=' ', mode='a')


def tag_person_entities(line):
    for sentence in sent_tokenize(line):
        count_persons = 0
        doc = nlp(sentence)

        for ent in doc.ents:
            if ent.label == PERSON:
                count_persons += 1

        if count_persons > 1:
            return True
        else:
            for token in sent_tokenize(sentence):
                if token in relationship_list:
                    return True
            return False


person_convs = ''
with open('../data/Friends_TV_Corpus/friends_conversations.txt', 'r') as f:
    for line in f.readlines():
        if tag_person_entities(line):
            person_convs += line

with open('../data/Friends_TV_Corpus/friends_conversation_lines_with_mult_persons_or_rels.txt', 'a') as f:
    f.write(person_convs)
