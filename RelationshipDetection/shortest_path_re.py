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


def extract_person_entities(sentence):
    """
    PER-PER
    :param sentence:
    :return:
    """
    doc = nlp(sentence)
    entities = []

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            entities.append(ent.text.lower())

    return entities


def extract_me_rel_entities(sentence):
    """
    ME-REL, REL-ME
    :param sentence:
    :return:
    """
    doc = nlp(sentence)
    entities = []
    rel = False

    for token in doc:
        if token.text.lower() in me_list:
            entities.append(token.text.lower())
        elif token.text.lower() in relationship_list:
            entities.append(token.text.lower())
            rel = True

    if not rel:
        entities = []

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

            # avoids 'me-me' or 'rel-rel' relations
            if not i == j and second_entity not in me_list and first_entity not in relationship_list:
                try:
                    shortest_path = nx.shortest_path(graph, source=first_entity, target=second_entity)
                    key = first_entity + '-' + second_entity
                    if len(shortest_path[1:-1]) > 0:
                        # path_dict[key] = shortest_path  # include entities in sp
                        path_dict[key] = shortest_path[1:-1]  # exclude entities in sp
                    else:
                        path_dict[key] = []
                except NodeNotFound as err:
                    logging.warning(f'Node not found: {err}')
                except NetworkXNoPath as err:
                    logging.warning(f'No path found: {err}')

    return path_dict


def measure_sp_rel_similarity(shortest_path):
    """
    :param shortest_path: dict of sp values
    :return:
    """

    highest_score = 0
    highest_rel = None

    for rel in relationship_list:
        try:
            score = model.n_similarity(shortest_path, [rel])

            if score > highest_score:
                highest_score = score
                highest_rel = rel
        except KeyError as err:
            logging.warning(err)

    if highest_score < 0.5:
        relation = 'KNOWS'
    else:
        relation = highest_rel

    return relation


def write_to_file(out_file, out_data):
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write(out_data)


def extract_relation_type(sp_dict, me_rel=False):
    extracted_relations = []
    for key, value in sp_dict.items():
        e1 = key.split('-')[0]
        e2 = key.split('-')[1]
        if len(value) > 0 and not me_rel:
            rel = measure_sp_rel_similarity(value)
            extracted_relation = str(f'<{e1, rel, e2}>')
        elif e2 in relationship_list:
            extracted_relation = str(f'<{e1, e2}>')
        elif e1 in relationship_list:
            extracted_relation = str(f'<{e2, e1}>')
        else:
            extracted_relation = str(f'<{e1},KNOWS, {e2}>')

        extracted_relations.append(extracted_relation)

    return extracted_relations


def extract_relations(text):
    extracted_relations = None

    for sentence in sent_tokenize(text):
       # print(f'##> Processing sentence: "{sentence}"')
        person_entities = extract_person_entities(sentence)
        me_rel_entities = extract_me_rel_entities(sentence)

        if len(person_entities) > 1:  # PER-PER
            #print(f'Entities found: {entities}')
            paths = search_shortest_dep_path(person_entities, sentence)
            extracted_relations = extract_relation_type(paths)
        elif len(person_entities) > 0 and len(me_rel_entities) > 0:  # USER-PER
            paths = search_shortest_dep_path(me_rel_entities + person_entities, sentence)
            extracted_relations = extract_relation_type(paths)
        elif len(me_rel_entities) > 1:  # USER-REL
            paths = search_shortest_dep_path(me_rel_entities, sentence)
            extracted_relations = extract_relation_type(paths, me_rel=True)

            """
            if paths:
                relations = measure_sp_rel_similarity(paths)
                for rel in relations:
                    relation_type = rel[1]

                    if relation_type:
                        e1 = rel[0].split('-')[0]
                        e2 = rel[0].split('-')[1]
                        extracted_relation = str(f'<{e1, relation_type, e2}>')
                        extracted_relations.append(extracted_relation)
            """
            # Lexanalyzer
            #lex = LexAnalyzer()
            #extracted_relation = lex.extract_rel(sentence)
            #if extracted_relation:
            #    extracted_relations.append(extracted_relation)

        else:
            print('no relations found')

    return extracted_relations


def extract_rels_from_convai(in_file, out_file):
    out_data = ''
    n = 0
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if n < 100:
                extracted = extract_relations(line)
                line = line.replace("\n", ";")
                if extracted:
                    out_data += f'{line} {extracted}\n'
                else:
                    out_data += f'{line} No relations found\n'
                n += 1
            else:
                break

    write_to_file(out_file, out_data)


utterance0 = u'''I have a son, he is 16 years old, and my dad, he is retired now.'''
utterance1 = u'''My daughter Lisa is moving to London next month.'''
utterance2 = u'''I've a son, he is in junior high school what are you doing for life?'''
utterance3 = u'''And my husband is a doctor I play in Baltimore Yeah.'''
utterance4 = u'''Peter and his brother Paul are walking to the beach.'''
utterance5 = u'''Peter, Steve and his brother Paul are walking to the beach.'''
utterance6 = u'''Me my angel twin sister love singing'''
utterance7 = u'''Hi my name is James'''
utterance8 = u'''i've a 9 year old son as well .'''


example_utterances = [utterance0, utterance1, utterance2, utterance3, utterance4, utterance5, utterance6, utterance7,
                      utterance8]
#extract_rels_from_convai(in_file='data/ConvAI2/extracted_conversations.txt',
#                         out_file='data/ConvAI2/extracted_relations.txt')

print(extract_relations(utterance1))


#for utterance in example_utterances:
#    print(utterance)
#    print(extract_relations(utterance))
#    print('\n')

