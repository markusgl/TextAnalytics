#import de_core_news_sm
import re
from nltk.tokenize import sent_tokenize
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

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
persons = ['Winnetou', 'Old Shatterhand', 'Sam Hawkens', 'Bloody Fox', 'Quick Panther', 'White Buffallo', 'Rollins']

tokenized_data = word_tokenize(clean_data)
# to check if tokenization works correct, print the first ten tokens
#print("Die ersten 10 Tokens: {}".format(tokenized_data[:10]))
#print("Gestamtanzahl Tokens: {}".format(len(tokenized_data)))

reltionship_dict ={}
word_window = ""
count = 0
for word in tokenized_data:
    word_window += word + ' '
    count += 1
    if count % 15 == 0:
        person_count = 0

        tmp_list = []
        for person in persons:
            if person in word_window:
                tmp_list.append(person)

        if len(tmp_list) > 1: # more than one person in window found
            for i in range(len(tmp_list)):
                dict_key = tmp_list[i] + "_" + tmp_list[i+1]
                if reltionship_dict[dict_key]:
                    reltionship_dict[dict_key] += 1
                else:
                    reltionship_dict[dict_key] = 1
        substring_words = ""

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
