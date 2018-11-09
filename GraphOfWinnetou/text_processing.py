"""
The Graph des Winnetou
"""

import re
import csv
import os

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import StanfordNERTagger
from GraphOfWinnetou.network_graph import NetworkGraph
from GraphOfWinnetou.neo4j_graph import Neo4jGraph

java_path = "C:/Program Files/Java/jdk1.8.0_181/bin/java.exe"
os.environ['JAVAHOME'] = java_path

# read the book
with open('./data/Winnetou_Band3.txt', 'r', encoding='utf-8') as f:
    raw_book_data = f.read()

# clean up book data
book_data = raw_book_data.replace('\n\n', ' ')  # remove blank lines
# skip the first the first 6 lines because they only contain meta info
headerlines_count = 0
clean_data = ''
for row in book_data.splitlines():
    if headerlines_count > 5:
        clean_data += str(row)
    headerlines_count += 1

clean_data = re.sub(r'\W', ' ', clean_data) # remove non-word characters
clean_data = re.sub(r'\s{2,}', ' ', clean_data)  # remove two or more consecutive whitespaces


mult_word_names = ['Fred Buller', 'Gustel Ebersbach', 'Ebersbachs Gustel', 'Don Fernando', 'Donna Elvira de Gonzalez',
                  'Donna Elvira', 'Elvira de Gonzalez', 'Sennor Henrico Gonzalez', 'Sennor Henrico', 'Henrico Gonzalez',
                   'Samuel Haller', 'Sam Hawkens', 'Vater Hillmann', 'alte Hillmann', 'jungen Hillmann',
                   'junge Hillmann', 'Frau Willys', 'Juwelier Marshall', 'Bernard Marshall', 'Bernard Marshall',
                   'Bill Meinert', 'Fred Morgan', 'Patrik Morgan', 'Old Shatterhand', 'Old Firehand', 'Pidas Squaw',
                   'Fred Walker']

# replace the whitespaces inside names consisting of multiple words with underscores
for name in mult_word_names:
    tokens = word_tokenize(name)
    replace_name = ""
    for i, token in enumerate(tokens):
        if i > 0:
            replace_name += '-' + token
        else:
            replace_name += token

    clean_data = re.sub(name, replace_name, clean_data)


"""
Extract every names using NER

persons_set = set()
nlp = de_core_news_sm.load()
for sentence in sent_tokenize(clean_data):
    doc = nlp(sentence)
    for ent in doc.ents:
        if ent.label_ == 'PER':
            #print(ent.text, ent.label_)
            persons_set.add(ent.text)

# TODO Test if Stanford NER works better
model = 'models/dewac_175m_600.crf.ser.gz'
# model = 'models/hgc_175m_600.crf.ser.gz'
st_ner = StanfordNERTagger(model,
                            'models/stanford-ner.jar',
                            encoding='utf-8')
person_set = set()
for sentence in sent_tokenize(book_data):
    classified = st_ner.tag(sentence)

    for entity in classified:
        entity_text = entity[0]
        entity_label = entity[1]
        if entity_label == 'I-PER':
            person_set.add(entity_text)

print(len(person_set))
print(person_set)
"""

# manually created list with persons
persons_dict = {'Alma': ['Alma'],
                'Bob': ['Bob'],
                'Buller': ['Fred-Buller', 'Buller'],
                'Capitano': ['Capitano'],
                'Clay': ['Clay'],
                'Conchez': ['Conchez'],
                'Auguste': ['Gustel-Ebersbach', 'Ebersbachs-Gustel', 'Ebersbach'],
                'Eulalia': ['Eulalia'],
                'Fernando': ['Don-Fernando'],
                'Gates': ['Gates'],
                'ElviraG.': ['Donna-Elvira-de-Gonzalez', 'Donna-Elvira', 'Elvira-de-Gonzalez'],
                'HenricG.': ['Sennor-Henrico-Gonzalez', 'Sennor-Henrico', 'Henrico-Gonzalez'],
                'Haller': ['Samuel-Haller', 'Haller'],
                'SamHawkens': ['Sam-Hawkens', 'Sam', 'Sams'],
                'Hi-Iah-dih': ['Hi-Iah-dih'],
                'V.Hillmann': ['Vater-Hillmann', 'alte-Hillmann', 'Hillmann'],
                'W.Hillmann': ['Willy', 'jungen-Hillmann', 'junge-Hillmann'],
                'Fr.HillmannJung': ['Frau-Willys'],
                'Hoblyn': ['Hoblyn'],
                'Holfert': ['Holfert'],
                'Inta': ['Inta'],
                'Kakho-oto': ['Kakho-oto'],
                'Ka-wo-mien': ['Ka-wo-mien'],
                'Ko-itse': ['Ko-itse'],
                'Ko-tu-cho': ['Ko-tu-cho'],
                'Ma-ram': ['Ma-ram'],
                'A.Marshall': ['Juwelier-Marshall'],
                'B.Marshall': ['Bernard-Marshall', 'Bernard-Marshall', 'Bernards', 'Marshall'],
                'Ma-ti-ru': ['Ma-ti-ru'],
                'B.Meinert': ['Bill-Meinert'],
                'F.Morgan': ['Fred-Morgan', 'Morgan'],
                'P.Morgan': ['Patrik-Morgan'],
                'Ohiamann': ['Ohiomann'],
                'Ohlers': ['Ohlers'],
                'OldShatterhand': ['Old-Shatterhand', 'Shatterhand'],
                'OldFirehand': ['Old-Firehand'],
                'Pida': ['Pida'],
                'PidasSquaw': ['Pidas-Squaw'],
                'Rudge': ['Rudge'],
                'Sanchez': ['Sanchez'],
                'Sans-ear': ['Sans-ear'],
                'Santer': ['Santer'],
                'Shelley': ['Shelley'],
                'Summer': ['Summer'],
                'Sus-Homascha': ['Sus-Homascha'],
                'Tangua': ['Tangua'],
                'Til-Lata': ['Til-Lata'],
                'To-kei-chun': ['To-kei-chun'],
                'F.Walker': ['Fred-Walker', 'Walker'],
                'Williams': ['Williams'],
                'Winnetou': ['Winnetou'],
                'Yato-Ka': ['Yato-Ka']
                }

relationship_dict = {}
tmp_list = []
tokenized_text = word_tokenize(clean_data)
substring_length = 10


def _search_persons(tokenized_string, primary_person=None):
    for key, value in persons_dict.items():
        for i, token in enumerate(tokenized_string):
            if token in value and key != primary_person:
                    dict_key = primary_person + '_' + key
                    if dict_key in relationship_dict.keys():
                        relationship_dict[dict_key] += 1
                    else:
                        relationship_dict[dict_key] = 1


def process_text():
    print("Tokens: {}".format(len(tokenized_text)))
    for key, value in persons_dict.items():
        for i, token in enumerate(tokenized_text):
            if token in value:
                found_person = key
                tmp_substring = tokenized_text[i - substring_length:i] + tokenized_text[i+1:i + substring_length]
                _search_persons(tmp_substring, found_person)


# save results to csv file
def save_to_csv(file_name):
    with open(file_name, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)

        csv_list = []
        for key, value in relationship_dict.items():
            if value > 2:  # only weights over 2 will be recognized as interaction
                tmp_list = key.split('_')
                tmp_list.append(str(value))
                csv_list.append(tmp_list)

        writer.writerows(csv_list)


# draw graph with networkx
def draw_graph(csv_file):
    ng = NetworkGraph()
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ng.add_edge(row[0], row[1], weight=int(row[2]))

    ng.draw_network()


# Save graph to neo4j database
def save_csv_to_neo4j(csv_file):
    neo4j_graph = Neo4jGraph()
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            first_node = neo4j_graph.get_node_by_name(str(row[0]))
            second_node = neo4j_graph.get_node_by_name(str(row[1]))

            if not first_node:
                first_node = neo4j_graph.add_node_by_name(str(row[0]))
            if not second_node:
                second_node = neo4j_graph.add_node_by_name(str(row[1]))

            neo4j_graph.add_relationship(first_node, second_node, weight=int(row[2]))

    neo4j_graph.add_pagerank()
    neo4j_graph.add_communites()


process_text()
save_to_csv('winnetou3_highranking.csv')
