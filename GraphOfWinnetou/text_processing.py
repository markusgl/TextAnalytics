"""
Der Graph des Manitou
"""

#import de_core_news_sm
import re
import csv

from nltk.tokenize import word_tokenize
from GraphOfWinnetou.network_graph import NetworkGraph
from GraphOfWinnetou.neo4j_graph import Neo4jGraph

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

#print(len(book_data))
#print(len(clean_data))

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
print(len(persons_set))
print(persons_set)
"""

# manually created list with persons
persons_dict = {'Alma': ['Alma'],
                'Bob': ['Bob'],
                'Buller': ['Fred Buller', 'Buller'],
                'Capitano': ['Capitano'],
                'Clay': ['Clay'],
                'Conchez': ['Conchez'],
                'Auguste': ['Gustel Ebersbach', 'Ebersbachs Gustel', 'Ebersbach'],
                'Eulalia': ['Eulalia'],
                'Fernando': ['Don Fernando'],
                'Gates': ['Gates'],
                'ElviraG.': ['Donna Elvira de Gonzalez', 'Donna Elvira', 'Elvira de Gonzalez'],
                'HenricG.': ['Sennor Henrico Gonzalez', 'Sennor Henrico', 'Henrico Gonzalez'],
                'Haller': ['Samuel Haller', 'Haller'],
                'SamHawkens': ['Sam Hawkens', 'Sam', 'Sams'],
                'Hi-Iah-dih': ['Hi-Iah-dih'],
                'V.Hillmann': ['Vater Hillmann', 'alte Hillmann', 'Hillmann'],
                'W.Hillmann': ['Willy', 'jungen Hillmann', 'junge Hillmann'],
                'F.HillmannJung': ['Frau Willys'],
                'Hoblyn': ['Hoblyn'],
                'Holfert': ['Holfert'],
                'Inta': ['Inta'],
                'Kakho-oto': ['Kakho-oto'],
                'Ka-wo-mien': ['Ka-wo-mien'],
                'Ko-itse': ['Ko-itse'],
                'Ko-tu-cho': ['Ko-tu-cho'],
                'Ma-ram': ['Ma-ram'],
                'A.Marshall': ['Juwelier Marshall'],
                'B.Marshall': ['Bernard Marshall', 'Bernard Marshall', 'Bernards', 'Marshall'],
                'Ma-ti-ru': ['Ma-ti-ru'], 'B.Meinert': ['Bill Meinert'],
                'F.Morgan': ['Fred Morgan', 'Morgan'],
                'P.Morgan': ['Patrik Morgan'],
                'Ohiamann': ['Ohiomann'],
                'Ohlers': ['Ohlers'],
                'OldShatterhand': ['Old Shatterhand'],
                'Pida': ['Pida'],
                'PidasSquaw': ['Pidas Squaw'],
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
                'F.Walker': ['Fred Walker', 'Walker'],
                'Williams': ['Williams'],
                'Winnetou': ['Winnetou'],
                'Yato-Ka': ['Yato-Ka']
                }

relationship_dict = {}
tmp_list = []
tokenized_data = word_tokenize(clean_data)
substring_length = 10


def search_persons(tokenized_string, primary_person):
    for i in range(len(tokenized_string)):
        for key, value in persons_dict.items():
            if tokenized_string[i] in value and key != primary_person:
                dict_key = primary_person + '_' + key
                if dict_key in relationship_dict.keys():
                    relationship_dict[dict_key] += 1
                else:
                    relationship_dict[dict_key] = 1
            elif i < len(tokenized_string)-1:
                if tokenized_string[i] + tokenized_string[i + 1] in value and key != primary_person:
                    dict_key = primary_person + '_' + key
                    if dict_key in relationship_dict.keys():
                        relationship_dict[dict_key] += 1
                    else:
                        relationship_dict[dict_key] = 1

# TODO recursive call

print("Tokens: {}".format(len(tokenized_data)))
for i in range(len(tokenized_data)):
    for key, value in persons_dict.items():
        if tokenized_data[i] in value:
            found_person = key
            tmp_substring = tokenized_data[i - substring_length:i] + tokenized_data[i:i + substring_length]
            search_persons(tmp_substring, found_person)
        elif i < len(tokenized_data)-1:
            if tokenized_data[i] + tokenized_data[i+1] in value:
                found_person = key
                tmp_substring = tokenized_data[i - substring_length:i] + tokenized_data[i:i + substring_length]
                search_persons(tmp_substring, found_person)


print(relationship_dict)


#search through tokenized text - OLD VERSION iterate once through tokenized data
"""
for i in range(len(tokenized_data)):
    if tokenized_data[i] in persons:
        found_person = tokenized_data[i]
        tmp_substring = tokenized_data[i-substring_length:i-1] + tokenized_data[i+1:i+substring_length]

        for word in tmp_substring:
            if word in persons and word != found_person:
                #ng.add_edge(found_person, word) # add to networkx
                dict_key = found_person + "_" + word
                if dict_key in relationship_dict:
                    relationship_dict[dict_key] += 1
                else:
                    relationship_dict[dict_key] = 1

#ng.draw_network()
#print(relationship_dict)
"""


# save results to csv file
def save_to_csv(file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        csv_list = []
        for key, value in relationship_dict.items():
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

            neo4j_graph.add_relationship(first_node, second_node)


file_name = 'test_graph.csv'
save_to_csv(file_name)
draw_graph(file_name)
