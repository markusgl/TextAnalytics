import spacy
import nltk
import pandas as pd
import re

from GraphOfWinnetou.neo4j_graph import Neo4jGraph

print('Loading spaCy model...')
nlp = spacy.load('de')
#text = u'''Meine kleine Enkelin Lisa und mein Enkel Lukas fliegen morgen nach London.'''
text = u'''Herbert ist der Vater von Hans'''
#text = u'''Peter und Maria gehen morgen ins Kino.'''

sentences = nltk.sent_tokenize(text)

feature_columns = ['ne', 'ne_type', 'ne_dep', 'ne_head']
features = pd.DataFrame(columns=feature_columns)

relationships = ['vater', 'mutter', 'sohn', 'tochter', 'bruder', 'schwester', 'enkel', 'enkelin', 'nichte',
            'neffe', 'onkel', 'tante']
#entities = ['lisa', 'max', 'hans', 'hubert', 'lukas', 'london', 'herbert', 'mein', 'meine']
me_entities = ['ich', 'mein', 'meine', 'meinen', 'meines']

for sentence in sentences:
    doc = nlp(sentence)

    tokens = []
    for token in doc:
        ne = token.text
        ne_dep = token.dep_
        head = token.head.text
        # pos = token.pos_
        # children = [child for child in token.children]
        # lemma = token.lemma_.lower()
        # print(token.text, token.dep_, token.head.text, token.head.pos_,
        # [child for child in token.children])
        data = {'ne': ne.lower(), 'ne_type': None, 'ne_dep': ne_dep, 'ne_head': head.lower()}
        training_ex = pd.Series(data, index=feature_columns)
        features = features.append(training_ex, ignore_index=True)

    for ent in doc.ents:
        features.loc[features['ne'] == ent.text.lower(), 'ne_type'] = ent.label_


neo4j_graph = Neo4jGraph()


def add_to_neo4j(node1, node2, node1_type, node2_type, rel_type):
    first_node = neo4j_graph.get_node_by_name(node1)
    second_node = neo4j_graph.get_node_by_name(node2)

    if not first_node:
        first_node = neo4j_graph.add_node_by_name(node1, node1_type)
    if not second_node:
        second_node = neo4j_graph.add_node_by_name(node2, node2_type)

    neo4j_graph.add_relationship(first_node, second_node, weight=1, name=rel_type)


rel_list = []

def iterate(exclude_elem, head):
    possible_rel = features[(features['ne_head'] == head) & ~features['ne'].isin([exclude_elem])
                                & ~features['ne_dep'].isin(['ROOT'])]

    # check if column 'ne' of possible_rel contains one or more named entities (real world entites)
    direct_rels = possible_rel[(possible_rel['ne_type'] == 'PER') | (possible_rel['ne'].isin(me_entities))
                                   & ~possible_rel['ne'].isin([exclude_elem])]

    if len(direct_rels) > 0:
        for ent in direct_rels.iterrows():
            entity = ent[1]['ne']

            if rel_list:
                relationship = [word for word in rel_list if word in relationships]
                if relationship:
                    print(f"({exclude_elem})-[{relationship}]->({entity})")
                else:
                    print(f"({exclude_elem})-['KNOWS']->({entity})")
            else:
                print(f"({exclude_elem})-[{head}]->({entity})")

            rel_list.clear()


    else:  # if no direct relationship between names was found iterate possible transitive rels
        for row in possible_rel.iterrows():
            entity = row[1]['ne']

            # if row[1]['ne_dep'] != 'root':  # look for transitive relationship
            rel_list.append(entity)
            iterate(exclude_elem, entity)


for i, row in enumerate(features['ne'].iteritems()):
    elem = row[1].lower()
    rel = []
    if elem in me_entities or features['ne_head'][i] == 'PER':
        head = features['ne_head'][i].lower()
        # print(f'{elem} {head}')
        iterate(elem, head)


