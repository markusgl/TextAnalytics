import networkx as nx
import spacy
import logging
import matplotlib.pyplot as plt
import re

from networkx.exception import NodeNotFound, NetworkXNoPath
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize

from flair.data import Sentence
from flair.models import SequenceTagger

from RelationshipDetection.lex_analyzer import LexAnalyzer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

nlp = spacy.load('en')
model = KeyedVectors.load_word2vec_format('../../Data/word_embeddings/GoogleNews-vectors-negative300.bin',
                                          binary=True, limit=30000)

relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'husband', 'wife']
me_list = ['i', 'my']
#tagger = SequenceTagger.load('ner')  #EN
tagger = SequenceTagger.load('de-ner')  #EN


def extract_pronoun_entities(text):
    entities = []
    for token in word_tokenize(text):
        if token.lower() in me_list:
            entities.append('USER')

    return entities


def extract_entities_flair(raw_text):
    entities = extract_pronoun_entities(raw_text)

    #raw_text = re.sub(r'\W+', ' ', raw_text)  # delete non word characters
    raw_text = re.sub('\s{2,}', ' ', raw_text)  # delete multiple consecutive spaces

    sentence = Sentence(raw_text.lower())  # instantiate sentence object
    tagger.predict(sentence)

    # NER Spans
    for entity in sentence.get_spans('ner'):
        if len(entity.tokens) > 1:
            entities.append(str(entity.text).replace(' ', '_'))
        else:
            entities.append(entity.text)

    return entities


def extract_entities(sentence):
    """
    extracts PER-PER and USER-PER entities
    :param sentence:
    :return:
    """
    doc = nlp(sentence)
    entities = []

    for token in doc:
        if token.text.lower() in me_list:
            entities.append('USER')

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            entities.append(ent.text.lower())

    return entities


def build_undirected_graph(sentence, plot=False):
    doc = nlp(sentence)
    edges = []
    for token in doc:
        for child in token.children:
            source = token.lower_
            sink = child.lower_
            if source in me_list:
                source = 'USER'
            elif sink in me_list:
                sink = 'USER'

            edges.append((f'{source}',
                          f'{sink}'))

    graph = nx.Graph(edges)
    #di_graph = nx.DiGraph(edges)

    if plot:
        plot_graph(graph)

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


def search_shortest_dep_path(entities, sentence, plot_graph):
    path_dict = {}
    graph = build_undirected_graph(sentence, plot=plot_graph)

    for i, first_entity in enumerate(entities):
        first_entity = first_entity.split('_')[0]  # use only first name of multi-word entities

        #for j in range(len(entities)):  # bidirectional relations
        for j in range(i+1, len(entities)):  # unidirectional relations
            second_entity = entities[j]
            second_entity = second_entity.split('_')[0]  # use only first name of multi-word entities

            #if not i == j and second_entity not in me_list and first_entity not in relationship_list:
            if not i == j and not first_entity == second_entity:
                try:
                    shortest_path = nx.shortest_path(graph, source=first_entity, target=second_entity)
                    key = first_entity + '-' + second_entity
                    if len(shortest_path[1:-1]) > 0:
                        # path_dict[key] = shortest_path  # include entities in sp
                        path_dict[key] = shortest_path[1:-1]  # exclude entities in sp
                    else:
                        path_dict[key] = []
                except NodeNotFound as err:
                    logger.warning(f'Node not found: {err}')
                except NetworkXNoPath as err:
                    logger.warning(f'No path found: {err}')

    return path_dict


def measure_sp_rel_similarity(shortest_path):
    """
    :param shortest_path: dict of sp values
    :return:
    """
    relation = None
    highest_score = 0
    highest_rel = None

    for rel in relationship_list:
        try:
            score = model.n_similarity(shortest_path, [rel])

            if score > highest_score:
                highest_score = score
                highest_rel = rel
        except KeyError as err:
            logger.debug(err)

    if highest_score > 0.5:
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
        #if len(value) > 0 and not me_rel:
        if len(value) > 0:
            rel = measure_sp_rel_similarity(value)
            if rel:
                extracted_relation = e1, rel, e2
                extracted_relations.append(extracted_relation)
        #elif e2 in relationship_list:
        #    extracted_relation = e1, e2
        #    rel_type = e2
        #elif e1 in relationship_list:
        #    extracted_relation = e2, e1
        #    rel_type = e1
        #else:
        #    extracted_relation = e1, 'KNOWS', e2

    return extracted_relations


def extract_relations(text, plot_graph=False):
    extracted_relations = []

    for sentence in sent_tokenize(text):
       # print(f'##> Processing sentence: "{sentence}"')
        person_entities = extract_entities(sentence)
        #me_rel_entities = extract_me_rel_entities(sentence)

        #if len(person_entities) > 0 and len(me_rel_entities) > 0:  # USER-PER
        #    logging.debug(f'Extracted entities: {person_entities + me_rel_entities}')
        #    paths = search_shortest_dep_path(me_rel_entities + person_entities, sentence)
        #    extracted_relations, rel_type = extract_relation_type(paths)

        logger.info(f'Extracted entities: {person_entities}')
        if len(person_entities) > 1:  # PER-PER or USER-PER
            paths = search_shortest_dep_path(person_entities, sentence, plot_graph)
            extracted_relations = extract_relation_type(paths)
        #elif len(me_rel_entities) > 1:  # USER-REL
        #    paths = search_shortest_dep_path(me_rel_entities, sentence)
        #    extracted_relations, rel_type = extract_relation_type(paths, me_rel=True)

        if len(extracted_relations) < 1:
            # USER-REL
            #logger.info('no relations found')
            # Lexanalyzer
            lex = LexAnalyzer()
            extracted_relation = lex.extract_rel(sentence)
            if extracted_relation:
                extracted_relations.append(extracted_relation)

    return extracted_relations


def extract_rels_from_convai(in_file, out_file):
    out_data = ''
    line_count = 0
    limit = 1000

    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line_count < limit:
                extracted = extract_relations(line)
                line = line.replace("\n", ";")
                if extracted:
                    out_data += f'{line} {extracted}\n'
                else:
                    out_data += f'{line} No relations found\n'
                line_count += 1
            else:
                break

    write_to_file(out_file, out_data)


def run_example_utterances(example_utterances):

    for utterance in example_utterances:
        print('>> ' + utterance)
        print(extract_relations(utterance))
        print('\n')


#extract_rels_from_convai(in_file='data/validation/convai2/validation_set.txt',
#                         out_file='data/validation/validation_set_extracted_relations_final.csv')


utterance0 = u'''I have a son, he is 16 years old, and my dad, he is retired now.'''
utterance1 = u'''My daughter Lisa is moving to London next month.'''
utterance2 = u'''I've a son, he is in junior high school what are you doing for life?'''
utterance3 = u'''And my husband is a doctor I play in Baltimore Yeah.'''
utterance4 = u'''Peter and his brother Paul are walking to the beach.'''
utterance5 = u'''Peter, Steve and his brother Paul are walking to the beach.'''
utterance6 = u'''Me my angel twin sister love singing'''
utterance7 = u'''Hi my name is James'''
utterance8 = u'''i've a 9 year old son as well .'''
utterance9 = u'''Maria and her brother Max are going to School'''
utterance10 = u'''i'm a call of duty girl i cant wait for the new one my younger brother Tom and his sister Jessica is a cod player too .'''
utterance11 = u'''My younger brother tom and his sister lisa simpson are cod player.'''
utterance12 = u'Meine kleine Enkelin Lisa und mein Enkel Lukas fliegen morgen nach London.'

example_utterances = [utterance0, utterance1, utterance2, utterance3, utterance4, utterance5, utterance6, utterance7,
                      utterance8, utterance9, utterance10, utterance11]
#run_example_utterances(example_utterances)

print(extract_relations(utterance12))
