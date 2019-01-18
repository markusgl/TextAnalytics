import nltk
import re
import spacy

from nltk.tokenize import word_tokenize
nlp = spacy.load('de')
grammar = r"""
            PP: {<PRON><AUX><DET><ADJ>?<NOUN>}
            NP: {<DET><ADJ>?<NOUN><PROPN>*}            
            REL: {<PP>|<NP>}"""

grammar_en = r"""
            PP: {<PRON><VERB><DET><ADJ>?<NOUN>}
            NP: {<DET><ADJ>?<NOUN><PROPN>*}            
            REL: {<PP>|<NP>}"""


relationship_list = ['vater', 'mutter', 'papa', 'papi', 'mama', 'mami', 'sohn', 'tochter', 'bruder', 'schwester',
                     'enkel', 'enkelin', 'nichte', 'neffe', 'großvater', 'großmutter', 'opa', 'opa',
                     'onkel', 'tante', 'cousin', 'cousine', 'schwager', 'schwägerin', 'Mann', 'Frau']
me_list = ['ich', 'mich', 'meine', 'meiner', 'mein']


def search_rel_type(sentence):
    rel_type = False

    for token in word_tokenize(sentence):
        if token.lower() in relationship_list:
            rel_type = True

    return rel_type


def pos_tag_sentence(sentence):
    sentence = re.sub('\W+', ' ', sentence)
    doc = nlp(sentence)

    pos_tagged_sentence = []
    for token in doc:
        pos_tuple = (token.text, token.pos_)
        pos_tagged_sentence.append(pos_tuple)

    return pos_tagged_sentence


def chunk_sentence(pos_tagged_sentence):
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tagged_sentence)

    return result


def extract_rel(sentence):
    me = None
    relative = None

    if search_rel_type(sentence):
        chunk = chunk_sentence(pos_tag_sentence(sentence))

        if chunk[0][0][0][0].lower() in me_list:
            me = chunk[0][0][0][0]
            relative = chunk[0][0][-1][0]

    return me, relative


# DE
positive_text1 = u'''Meine kleine Schwester'''
positive_text2 = u'''Mein Schwester'''
positive_text3 = u'''Mein Bruder Karl'''
positive_text4 = u'''Ich habe einen Bruder'''
positive_text5 = u'''Ich habe einen älteren Bruder'''

negative_text1 = u'''Ihr Bruder'''
negative_text2 = u'''sein Vater Peter'''
negative_text3 = u'''die Brüder von Lisa'''
negative_text4 = u'''Ich arbeite als Barkeeper'''
negative_text5 = u'''Meine Katze ist ziemlich dick'''
negative_text6 = u'''Ich habe drei Kinder'''

# EN
positive_text1_en = u'''My little sister'''
positive_text2_en = u'''I have an older brother'''

negative_text1_en = u'''her brother'''
negative_text2_en = u'''his father John'''

sentences_pos = [positive_text1, positive_text2, positive_text3, positive_text4, positive_text5]
sentences_neg = [negative_text1, negative_text2, negative_text3, negative_text4, negative_text5, negative_text6]

print('## Positive')
for sentence in sentences_pos:
    print(extract_rel(sentence))

print('\n## Negative')
for sentence in sentences_neg:
    print(extract_rel(sentence))

