"""
Der Graph des Manitou
"""

#import de_core_news_sm
import re
from nltk.tokenize import word_tokenize
from network_graph import NetworkGraph

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
persons = ['Charles', 'Capitano', 'Bob', 'Alma', 'Buller', 'Clay', 'Charley', 'Tony', 'Summer', 'Conchez', 'Ebersbach',
           'Eulalia', 'Don Fernando', 'Fernando', 'Gates', 'Elvira', 'Gonzalez', 'Henrico', 'Gonzalez', 'Samuel', 'Haller',
           'Hi-lah-dih', 'Willy', 'Hillmann', 'Hoblyn', 'Holfert', 'Inta', 'Kakho-oto', 'Ka-wo-mien', 'Ko-itse',
           'Ko-tu-cho', 'Ma-ram', 'Allan', 'Bernard', 'Ma-ti-ru', 'Bill', 'Meinert', 'Fred', 'Morgan', 'Patrik',
           'Morgan', 'Ohiomann', 'Ohlers', 'Firehand', 'Shatterhand', 'Pida', 'Pidas', 'Squaw', 'Rollins', 'Rudge',
           'Hawkens', 'Sanchez', 'Sans-ear', 'Santer', 'Shelley', 'Summer', 'Sus-Homascha', 'Tangua', 'Til-Lata',
           'To-kei-chun', 'Walker', 'Williams', 'Winnetou', 'Yato-Ka']


relationship_dict = {}
tmp_list = []
tokenized_data = word_tokenize(clean_data)
substring_length = 10

ng = NetworkGraph()

# generate 15 words substring
for i in range(len(tokenized_data)):
    if tokenized_data[i] in persons:
        found_person = tokenized_data[i]
        tmp_substring = tokenized_data[i-substring_length:i-1] + tokenized_data[i+1:i+substring_length]

        for word in tmp_substring:
            if word in persons:
                ng.add_edge(found_person, word) #
                dict_key = found_person + "_" + word
                if dict_key in relationship_dict:
                    relationship_dict[dict_key] += 1
                else:
                    relationship_dict[dict_key] = 1

ng.draw_network()
print(relationship_dict)


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
