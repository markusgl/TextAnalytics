""" playing with Zalando's open source NLP framework flair https://github.com/zalandoresearch/flair """

import re
import torch

from flair.data import Sentence, Token
from flair.embeddings import WordEmbeddings

text = u'Hans, welcher der Sohn von Hubert ist, geht mit Peter ins Kino.'
text2 = u'kino sohn ist peter von karl'

# remove special characters
text = re.sub(r'\W', ' ', text)
text = re.sub(r'\s{2,}', ' ', text)  # remove multiple following whitespaces

sentence = Sentence(text.lower())
sentence2 = Sentence(text2)

glove_embedding = WordEmbeddings('de')
#glove_embedding = WordEmbeddings('de-crawl')
glove_embedding.embed([sentence, sentence2])




tensors = []
for token in sentence:
    if token.text == 'kino':
        tensors.append(token.embedding)

for token in sentence2:
    if token.text == 'kino':
        tensors.append(token.embedding)

# compute equality of tensors

for i in range(len(tensors)):
    for j in range(len(tensors)):
        print(torch.equal(tensors[i], tensors[j]))
        


