import spacy

from spacy.symbols import PERSON
from nltk.tokenize import sent_tokenize

nlp = spacy.load('en')

relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'mommy', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'grampa', 'grandpa', 'grandma', 'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'brother-in-law',
                     'sister-in-law', 'husband', 'wife']


def search_entity_or_relation(sentence):
    doc = nlp(sentence)
    extract = False

    for ent in doc.ents:
        if ent.label == PERSON:
            extract = True

    for token in doc:
        if token.text.lower() in relationship_list:
            extract = True

    return extract


with open('../data/ConvAI2/train_none_original_no_cands.txt', 'r') as f:
    corpus = ''
    data = f.readlines()
    for line in data:
        for sentence in sent_tokenize(line[2:]):
            if search_entity_or_relation(sentence):
                corpus += sentence + '\n'

# out file
with open('../data/ConvAI2/extracted_conversations_with_relations.txt', 'w') as f:
    f.write(corpus)

