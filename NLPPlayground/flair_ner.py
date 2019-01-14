""" playing with Zalando's open source NLP framework flair https://github.com/zalandoresearch/flair """

import re
from nltk.tokenize import sent_tokenize
from flair.data import Sentence
from flair.models import SequenceTagger

# load the NER tagger
tagger = SequenceTagger.load('de-ner')
named_entities = []


def tag_person_entities(raw_text):
    raw_text = re.sub(r'\W+', ' ', raw_text)
    sentence = Sentence(raw_text)  # instantiate sentence object
    tagger.predict(sentence)

    print('\n##### NER SPANS #####')
    # NER Spans
    for entity in sentence.get_spans('ner'):
        named_entities.append(entity)
        print(entity)

    print('\n##### TAG EACH TOKEN #####')
    # NER Tags for each word
    for token in sentence:
        ner_tag = token.get_tag('ner')
        print(f'{token}, {ner_tag}')


def ner_book_data(filename):
    file_name = filename

    with open('../RelationshipDetection/data/'+file_name+'.txt', 'r', encoding='utf-8') as f:
        data = f.read()

    for line in sent_tokenize(data):
        tag_person_entities(line)

    print(named_entities)


text = 'Hans Müller und sein Sohn Hubert fliegen nach New York. '
text2 = u'Meine kleine Enkelin Lisa und mein Enkel Lukas fliegen morgen nach London.'
text3 = u'''Hans, welcher der Sohn von Hubert ist, geht mit Peter ins Kino.'''

tag_person_entities(text)
#ner_book_data('Robinson_Crusoe')

#print(named_entities)
