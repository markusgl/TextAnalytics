"""
Der Graph des Manitou
"""

#import de_core_news_sm
import re
import csv

from nltk.tokenize import word_tokenize
from network_graph import NetworkGraph
from neo4j_graph import Neo4jGraph

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
persons = ['Alma', 'Bob', 'Buller', 'Capitano', 'Clay', 'Conchez', 'Conchez', 'Ebersbach', 'Eulalia', 'Fernando',
            'Gates', 'Elvira', 'Henrico', 'Haller', 'Hawkens', 'Hi-lah-dih', 'Hillmann', 'Hoblyn', 'Holfert',
            'Inta', 'Kakho-oto', 'Ka-wo-mien', 'Ko-itse', 'Ko-tu-cho', 'Ma-ram', 'Allan', 'Bernard', 'Ma-ti-ru',
            'Meinert', 'Fred', 'Patrik', 'Ohiomann', 'Firehand', 'Shatterhand', 'Pida', 'Squaw', 'Rollins',
            'Sanchez', 'Sans-ear', 'Santer', 'Shelley', 'Summer', 'Tony', 'Sus-Homascha', 'Tangua', 'Til-Lata',
            'To-kei-chun', 'Walker', 'Williams', 'Winnetou', 'Yato-Ka']


relationship_dict = {}
tmp_list = []
tokenized_data = word_tokenize(clean_data)
substring_length = 10



# generate 15 words substring
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

with open('winnetou3_persons.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    csv_list = []
    for key, value in relationship_dict.items():
        tmp_list = key.split('_')
        tmp_list.append(str(value))
        csv_list.append(tmp_list)

    writer.writerows(csv_list)

# Draw graph
ng = NetworkGraph()
neo4j_graph = Neo4jGraph()
with open('winnetou3_persons.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        print(row)
        ng.add_edge(row[0], row[1], weight=int(row[2]))

        first_node = neo4j_graph.get_node_by_name(str(row[0]))
        second_node = neo4j_graph.get_node_by_name(str(row[1]))

        if not first_node:
            first_node = neo4j_graph.add_node_by_name(str(row[0]))
        if not second_node:
            second_node = neo4j_graph.add_node_by_name(str(row[1]))

        neo4j_graph.add_relationship(first_node, second_node)


ng.draw_network()


"""
Sliding window of 15 words

# test if it works
for i in range(10):
    tmp_list = tokenized_data[i:i+15]
    print(tmp_list)

relationship_dict = {}
for i in range(len(tokenized_data)):
    tmp_list = tokenized_data[i:i+20]
    #[token for token in tmp_list if token in persons]
    person_list = []
    for token in tmp_list:
        if token in persons:
            person_list.append(token)

    if len(person_list) > 1:
        relationship_dict[person_list[0]] = person_list[1]

print(relationship_dict)
"""
