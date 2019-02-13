""" playing with Zalando's open source NLP framework flair https://github.com/zalandoresearch/flair """

import re
import torch

from flair.data import Sentence, Token
from flair.embeddings import WordEmbeddings, FlairEmbeddings

text = u'Hans, welcher der Sohn von Hubert ist, geht mit Peter ins Kino.'

# remove special characters
text = re.sub(r'\W', ' ', text)
text = re.sub(r'\s{2,}', ' ', text)  # remove multiple following whitespaces


#glove_embedding = WordEmbeddings('de')
#glove_embedding = WordEmbeddings('de-crawl')
glove_embedding = WordEmbeddings('en-glove')


def extract_entities(sentence):
    glove_embedding.embed(sentence)
    for token in sentence:
        print(token)
        print(token.embedding)


def append_tensors(sentence):
    tensors = []
    for token in sentence:
        if token.text == 'kino':
            tensors.append(token.embedding)


    return tensors


def similarity(tensors):
    # compute equality of tensors
    for i in range(len(tensors)):
        for j in range(len(tensors)):
            print(torch.equal(tensors[i], tensors[j]))


text1 = 'Rose is the grandma of Lisa.'
sentence = Sentence(text1.lower())
extract_entities(sentence)

from flair.embeddings import BertEmbeddings

# init embedding
embedding = BertEmbeddings()

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)