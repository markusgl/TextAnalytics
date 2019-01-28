## LexAnalyzer based on ReVerb (Etzioni et al. 2008)

import spacy
import re
import nltk
from nltk.tokenize import word_tokenize

relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'husband', 'wife']
me_list = ['i', 'my']

# PP: e.g. 'I have a son', 'I have a smaller brother', 'I have a 9 year old son'
# NP: e.g. 'My (little) sister (Lisa)'
grammar = r"""
            PP: {<PRON><VERB><DET><ADJ>?<NOUN>}
            NP: {<ADJ><ADJ>?<NOUN><PROPN>*}            
            REL: {<PP>|<NP>}"""


class LexAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en')

    def search_rel_type(self, sentence):
        for token in word_tokenize(sentence):
            if token.lower() in relationship_list:
                return token.lower()

        return None

    def pos_tag_sentence(self, sentence):
        sentence = re.sub('\W+', ' ', sentence)
        doc = self.nlp(sentence)

        pos_tagged_sentence = []
        for token in doc:
            pos_tuple = (token.text, token.pos_)
            pos_tagged_sentence.append(pos_tuple)

        return pos_tagged_sentence

    def chunk_sentence(self, pos_tagged_sentence, draw=False):
        cp = nltk.RegexpParser(grammar)
        result = cp.parse(pos_tagged_sentence)

        if draw:
            result.draw()

        return result

    def extract_rel(self, sentence):
        extracted_relations = []

        # build chunks
        chunk_tree = self.chunk_sentence(self.pos_tag_sentence(sentence))

        for i, sub_tree in enumerate(chunk_tree):
            if type(sub_tree) is nltk.tree.Tree and sub_tree.label() == 'REL':
                me = sub_tree[0][0][0].lower()
                rel = [word for word in sub_tree[0] if word[0] in relationship_list]
                if me in me_list and rel:
                    relation = [item for item in rel[0]]

                    if sub_tree[0][-1][1] == 'PROPN':
                        rel_person = sub_tree[0][-1][0]
                        extracted_relations.append(f'<USER, {relation[0]}, {rel_person}>')
                    else:
                        extracted_relations.append(f'<USER, {relation[0]}>')

        return extracted_relations
