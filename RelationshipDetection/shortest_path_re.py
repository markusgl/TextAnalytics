import enum
import networkx as nx
import spacy
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
import re

from networkx.exception import NodeNotFound, NetworkXNoPath
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize
from flair.data import Sentence, Token
from flair.embeddings import WordEmbeddings

nlp = spacy.load('de')
model = KeyedVectors.load_word2vec_format('../models/german.model', binary=True)


relationship_list = ['vater', 'mutter', 'sohn', 'tochter', 'bruder', 'schwester', 'enkel', 'enkelin', 'nichte',
                     'neffe', 'onkel', 'tante']
me_list = ['ich', 'meine', 'mein', 'meiner', 'meinem', 'meinen']


def create_flair_embeddings(text):
    flair_embeddings = {}
    for sentence in sent_tokenize(text):
        sentence = re.sub(r'\W', ' ', sentence)
        sentence = re.sub(r'\s{2,}', ' ', sentence)

        sentence = Sentence(sentence.lower())
        glove_embedding = WordEmbeddings('de')
        # glove_embedding = WordEmbeddings('de-crawl')
        glove_embedding.embed(sentence)

        for token in sentence:
            flair_embeddings[token.text] = token.embedding

    return flair_embeddings



from flair.data import Sentence
from flair.models import SequenceTagger

def extract_entities(raw_sentence):
    entities = []

    clean_sentence = re.sub('\W+', ' ', raw_sentence)  # remove non-word characters
    sentence = Sentence(clean_sentence)
    tagger = SequenceTagger.load('de-ner')
    tagger.predict(sentence)  # run NER over sentence

    # NER spans
    print('Trying to extract entities...')

    for entity in sentence.get_spans('ner'):
        print(f'Entity: {entity}')

        if entity.tag == 'PER':
            if len(entity.tokens) > 1:  # if it is a multi word entity, replace blanks with underscores
                entities.append(str(entity.text.lower()).replace(' ', '_'))
            else:
                entities.append(entity.text.lower())


def extract_features(sp_dict):
    feature_columns = ['m1', 'm2', 'short_path']
    features = pd.DataFrame(columns=feature_columns)

    for key, value in sp_dict.items():
        m1 = key.split('-')[0]
        m2 = key.split('-')[1]
        short_path = value

        data = {'m1': m1, 'm2': m2, 'short_path': short_path}

        training_example = pd.Series(data, index=feature_columns)
        features = features.append(training_example, ignore_index=True)

    return features


def find_shortest_path(entities, graph):
    sp_dict = {}
    for i, first_entity in enumerate(entities):
        for j in range(i + 1, len(entities)):
            second_entity = entities[j]
            if not i == j and second_entity not in me_list:
                try:
                    shortest_path = nx.shortest_path(graph, source=first_entity, target=second_entity)
                    key = first_entity + '-' + second_entity
                    sp_dict[key] = shortest_path
                except NodeNotFound as err:
                    logging.warning(f'Node not found: {err}')
                except NetworkXNoPath as err:
                    logging.warning(f'No path found: {err}')

    return sp_dict


def plot_graph(graph):
    pos = nx.spring_layout(graph)
    # nodes
    nx.draw_networkx_nodes(graph, pos, node_size=200)
    # edges
    nx.draw_networkx_edges(graph, pos, width=1)
    # labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')

    plt.axis('off')  # disable axis
    plt.show()


def build_undirected_graph(doc):
    edges = []
    for token in doc:
        for child in token.children:
            edges.append((f'{token.lower_}',
                          f'{child.lower_}'))

    graph = nx.Graph(edges)
    plot_graph(graph)

    return graph


def recognize_named_entities(doc):
    entities = []
    for ent in doc.ents:
        if ent.label_ == 'PER':
            entities.append(ent.text.lower())

    for token in doc:
        if token.text.lower() in me_list:
            entities.append(token.text.lower())

    return entities


def extract_relationships(text):
    for sentence in sent_tokenize(text):
        doc = nlp(sentence)
        graph = build_undirected_graph(doc)
        entities = recognize_named_entities(doc)
        sp_dict = find_shortest_path(entities, graph)
        extract_features(sp_dict)


