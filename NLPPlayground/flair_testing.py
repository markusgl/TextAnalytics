""" testing Zalando's NLP framework flair https://github.com/zalandoresearch/flair """

from flair.data import Sentence
from flair.models import SequenceTagger

# make a sentence
sentence = Sentence('Hans und sein Sohn Hubert fliegen nach New York.')

# load the NER tagger
tagger = SequenceTagger.load('ner')

# run NER over sentence
tagger.predict(sentence)


print(sentence)
print('The following NER tags are found:')

# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)
