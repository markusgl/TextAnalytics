""" playing with Zalando's open source NLP framework flair https://github.com/zalandoresearch/flair """

from flair.data import Sentence
from flair.models import SequenceTagger

text = 'Hans und sein Sohn Hubert fliegen nach New York.'
text2 = u'Meine kleine Enkelin Lisa und mein Enkel Lukas fliegen morgen nach London.'

sentence = Sentence(text)

# load the NER tagger
tagger = SequenceTagger.load('ner')

# run NER over sentence
tagger.predict(sentence)

print(sentence)
print('The following NER tags are found:')

# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)
    print(f'ner-tag: {entity.tag}')

for token in sentence:
    ner_tag = token.get_tag('ner')
    print(f'{token}, {ner_tag}')

