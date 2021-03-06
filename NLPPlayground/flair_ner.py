""" playing with Zalando's open source NLP framework flair https://github.com/zalandoresearch/flair """

import re
from nltk.tokenize import sent_tokenize
from flair.data import Sentence
from flair.models import SequenceTagger

# load the NER tagger
#tagger = SequenceTagger.load('de-ner')  #DE
tagger = SequenceTagger.load('ner')  #EN
named_entities = []

me_list = ['i', 'my']


def tag_person_entities(raw_text):
    raw_text = re.sub(r'\W+', ' ', raw_text)
    sentence = Sentence(raw_text)  # instantiate sentence object
    tagger.predict(sentence)

    print('\n##### NER SPANS #####')
    # NER Spans

    for entity in sentence.get_spans('ner'):
        print(entity)
        if len(entity.tokens) > 1:
            named_entities.append(str(entity.text).replace(' ', '_'))
        else:
            named_entities.append(entity.text)


    per_dict = sentence.to_dict(tag_type='ner')
    print(per_dict)
    persons = []
    for dict in per_dict['entities']:
        person = dict['text']
        if len(person) > 1:
            persons.append(person.replace(' ', '_'))
        else:
            persons.append(person)

    print(persons)

    entities = ['']
    print('\n##### TAG EACH TOKEN #####')
    # NER Tags for each word
    for token in sentence:
        if token in me_list:
            entities.append(token.text.lower())

        ner_tag = token.get_tag('ner')
        print(f'{token}, {ner_tag}')


def ner_book_data(filename):
    file_name = filename

    with open('../RelationshipDetection/data/'+file_name+'.txt', 'r', encoding='utf-8') as f:
        data = f.read()

    for line in sent_tokenize(data):
        tag_person_entities(line)

    print(named_entities)


def basic_ner(text):
    # load model
    tagger = SequenceTagger.load('de-ner')

    # make German sentence
    sentence = Sentence(text)
    # predict NER tags
    tagger.predict(sentence)
    # print sentence with predicted tags
    print(sentence.to_tagged_string())


utterance1 = 'Hans Müller und sein Sohn Hubert fliegen nach New York. '
utterance2 = u'Meine kleine Enkelin Lisa und mein Enkel Lukas fliegen morgen nach London.'
utterance3 = u'''Hans, welcher der Sohn von Hubert ist, geht mit Peter ins Kino.'''
utterance4 = u'''Meine Schwester lebt in Hamburg.'''
utterance5 = u'''Meine Schwester Lisa lebt in Hamburg.'''
multiline_text = u'''Bart, welcher der Sohn von Homer ist, geht mit Milhouse ins Kino.
Meine kleine Enkelin Lisa und mein Enkel Bart fliegen morgen nach London. Ned Flanders ist der Vater von Rod und Todd.'''
utterance6 = u'''my younger brother tom and his sister lisa simpson is a cod player too.'''
utterance7 = u'''Victoria Andrew Kidman is the grandma of Lisa'''
utterance8 = u'''my sister , madonna , does too .'''
utterance9 = u'''my sister , bob marley , and my sister nicole kidman and my aunt madonna does too .'''

#tag_person_entities(utterance7)
#print(named_entities)

basic_ner('George Washington ging nach Washington .')
