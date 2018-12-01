from GraphOfWinnetou.neo4j_graph import Neo4jGraph
import spacy

relationship_list = ['vater', 'mutter', 'sohn', 'tochter', 'bruder', 'schwester', 'enkel', 'enkelin', 'nichte',
                     'neffe', 'onkel', 'tante']
me_list = ['ich', 'meine', 'mein', 'meinen', 'meines']

nlp = spacy.load('de')
#doc = nlp(u'''Herbert ist der Vater von Hans''')
#doc = nlp(u'Hans und sein Sohn Hubert')
#doc = nlp(u'Meine kleine Enkelin Lisa und mein Enkel Lukas fliegen morgen nach London')
doc = nlp(u'''Ich und mein Sohn gehen heute zum Fußball''')

neo4j_graph = Neo4jGraph()


def add_to_neo4j(node1, node2, node1_type, node2_type):
    if not node1_type == 'MISC':
        first_node = neo4j_graph.get_node_by_name(node1)
    if not node2_type == 'MISC':
        second_node = neo4j_graph.get_node_by_name(node2)

    if not first_node:
        first_node = neo4j_graph.add_node_by_name(node1, node1_type)
    if not second_node:
        second_node = neo4j_graph.add_node_by_name(node2, node2_type)

    neo4j_graph.add_relationship(first_node, second_node, weight=1)
    #neo4j_graph.add_pagerank()
    #neo4j_graph.add_communites()


rels = []
nodes = []
def iterate(text):
    for token in text:
        children = [child for child in token.children]
        if children:
            iterate(children)

        #print(token.text, token.dep_, token.head.text, token.pos_)
        rels.append((token.text, token.head.text))

        node1 = token.text
        node2 = token.head.text

        if node1.lower() in relationship_list:
            node1_type = 'RELATION'
        elif node1.lower() in me_list:
            node1 = 'me'
            node1_type = 'ME'
        elif token.pos_ == 'PROPN' or token.pos_ == 'NOUN':
            node1_type = 'ENTITY'
        else:
            node1_type = 'MISC'

        if node2.lower() in relationship_list:
            node2_type = 'RELATION'
        elif node2.lower() in me_list:
            node2 = 'me'
            node2_type = 'ME'
        elif token.head.pos_ == 'PROPN' or token.head.pos_ == 'NOUN':
            node2_type = 'ENTITY'
        else:
            node2_type = 'MISC'

        add_to_neo4j(node1, node2, node1_type, node2_type)


for token in doc:
    if token.dep_ == 'ROOT':
        children = [child for child in token.children]

        iterate(children)

print(f'RELS: {rels}')



#TODO change rel nodes to edges
