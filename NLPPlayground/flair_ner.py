""" playing with Zalando's open source NLP framework flair https://github.com/zalandoresearch/flair """

from nltk.tokenize import sent_tokenize
from flair.data import Sentence
from flair.models import SequenceTagger

# load the NER tagger
tagger = SequenceTagger.load('ner')
named_entities = []


def tag_person_entities(text):
    sentence = Sentence(text)
    tagger.predict(sentence)

    # iterate over entities and print
    # NER Spans
    for entity in sentence.get_spans('ner'):
        named_entities.append(entity)

    # NER Tags for each word
    #for token in sentence:
    #    ner_tag = token.get_tag('ner')
    #    print(f'{token}, {ner_tag}')


text = 'Hans und sein Sohn Hubert fliegen nach New York.'
text2 = u'Meine kleine Enkelin Lisa und mein Enkel Lukas fliegen morgen nach London.'
#tag_person_entities(text)

file_name = 'Robinson_Crusoe'
with open('../RelationshipDetection/data/'+file_name+'.txt', 'r', encoding='utf-8') as f:
    data = f.read()

for line in sent_tokenize(data):
    tag_person_entities(line)

print(named_entities)
