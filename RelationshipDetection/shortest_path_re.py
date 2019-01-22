import networkx as nx
import spacy
import logging
import matplotlib.pyplot as plt

from networkx.exception import NodeNotFound, NetworkXNoPath
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize

from RelationshipDetection.lex_analyzer import LexAnalyzer

nlp = spacy.load('en')
model = KeyedVectors.load_word2vec_format('../../Data/word_embeddings/GoogleNews-vectors-negative300.bin',
                                          binary=True, limit=30000)

relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'husband', 'wife']
me_list = ['i', 'my']


def tag_person_entities(sentence):
    doc = nlp(sentence)

    entities = []
    for token in doc:
        if token.text.lower() in me_list:
            entities.append(token.text.lower())

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            entities.append(ent.text.lower())

    return entities


def build_undirected_graph(sentence, plot=False):
    doc = nlp(sentence)
    edges = []
    for token in doc:
        for child in token.children:
            edges.append((f'{token.lower_}',
                          f'{child.lower_}'))
    graph = nx.Graph(edges)
    di_graph = nx.DiGraph(edges)

    if plot:
        plot_graph(di_graph)

    return graph


def plot_graph(graph):
    # nx.draw_networkx(graph, node_size=100, ode_color=range(len(graph)))
    pos = nx.spring_layout(graph)  # positions for all nodes
    # nodes
    nx.draw_networkx_nodes(graph, pos, node_size=200)
    # edges
    nx.draw_networkx_edges(graph, pos, width=1)
    # labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')

    plt.axis('off')  # disable axis
    plt.show()


def search_shortest_dep_path(entities, sentence):
    path_dict = {}
    graph = build_undirected_graph(sentence, plot=False)

    for i, first_entity in enumerate(entities):
        first_entity = first_entity.split('_')[0]  # use only first name of multi-word entities

        #for j in range(len(entities)):  # bidirectional relations
        for j in range(i+1, len(entities)):  # unidirectional relations
            second_entity = entities[j]
            second_entity = second_entity.split('_')[0]  # use only first name of multi-word entities

            if not i == j and second_entity not in me_list:
                try:
                    shortest_path = nx.shortest_path(graph, source=first_entity, target=second_entity)
                    key = first_entity + '-' + second_entity
                    if len(shortest_path[1:-1]) > 0:
                        # path_dict[key] = shortest_path  # include entities in sp
                        path_dict[key] = shortest_path[1:-1]  # exclude entities in sp
                    else:
                        path_dict[key] = ['KNOWS']
                        #return None
                except NodeNotFound as err:
                    logging.warning(f'Node not found: {err}')
                except NetworkXNoPath as err:
                    logging.warning(f'No path found: {err}')

    return path_dict


def measure_similarity(path_dict):
    relations = []
    for key, value in path_dict.items():
        if 'KNOWS' in value:
            relation = key, 'KNOWS'
        else:
            highest_score = 0
            highest_rel = None

            for rel in relationship_list:
                try:
                    score = model.n_similarity(value, [rel])

                    if score > highest_score:
                        highest_score = score
                        highest_rel = rel
                except KeyError as err:
                    logging.warning(err)

            if highest_score == 0:
                relation = key, 'KNOWS'
            else:
                relation = key, highest_rel

        relations.append(relation)

    return relations


def write_to_file(out_file, out_data):
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write(out_data)


def extract_relations(text):
    extracted_relations = []

    for sentence in sent_tokenize(text):
       # print(f'##> Processing sentence: "{sentence}"')
        entities = tag_person_entities(sentence)

        if len(entities) > 1:  # two or more persons found in sentence
            #print(f'Entities found: {entities}')
            paths = search_shortest_dep_path(entities, sentence)
            #print(paths)
            if paths:
                relations = measure_similarity(paths)
                for rel in relations:
                    relation_type = rel[1]

                    if relation_type:
                        e1 = rel[0].split('-')[0]
                        e2 = rel[0].split('-')[1]
                        extracted_relation = str(f'<{e1, relation_type, e2}>')
                        extracted_relations.append(extracted_relation)
        else:
            # Lexanalyzer
            lex = LexAnalyzer()
            extracted_relation = lex.extract_rel(sentence)
            if extracted_relation:
                extracted_relations.append(extracted_relation)

    return extracted_relations


def extract_rels_from_convai(in_file, out_file):
    out_data = ''
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            extracted = extract_relations(line)
            line = line.replace("\n", ";")
            if extracted:
                out_data += f'{line} {extracted}\n'
            else:
                out_data += f'{line} No relations found\n'

    write_to_file(out_file, out_data)


utterance = u'''I have a son, he is 16 years old, and my dad, he is retired now.'''
utterance1 = u'''My daughter Lisa is moving to London next month.'''
utterance2 = u'''I've a son, he is in junior high school what are you doing for life?'''
utterance3 = u'''And my husband is a doctor I play in Baltimore Yeah.'''
utterance4 = u'''Peter and his brother Paul are walking to the beach.'''
utterance5 = u'''Peter, Steve and his brother Paul are walking to the beach.'''

extract_rels_from_convai('data/convai/human_rel_sentences.txt', 'data/extracted_relations_new.txt')

#extract_relations(utterance1)
