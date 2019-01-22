import nltk
import re
import spacy

from nltk.tokenize import word_tokenize
nlp = spacy.load('en')
grammar = r"""
            PP: {<PRON><AUX><DET><ADJ>?<NOUN>}
            NP: {<DET><ADJ>?<NOUN><PROPN>}            
            REL: {<PP>|<NP>}"""

grammar_en = r"""
            PP: {<PRON><VERB><DET>?<ADJ>?<NOUN>}
            NP: {<ADJ><ADJ>?<NOUN><PROPN>?}            
            REL: {<PP>|<NP>}"""

"""
relationship_list = ['vater', 'mutter', 'papa', 'papi', 'mama', 'mami', 'sohn', 'tochter', 'bruder', 'schwester',
                     'enkel', 'enkelin', 'nichte', 'neffe', 'großvater', 'großmutter', 'opa', 'opa',
                     'onkel', 'tante', 'cousin', 'cousine', 'schwager', 'schwägerin', 'Mann', 'Frau']
me_list = ['ich', 'mich', 'meine', 'meiner', 'mein']
"""

relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'mommy', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'grampa', 'grandpa', 'grandma', 'niece', 'nephew', 'uncle', 'aunt', 'cousin'
                                                                                         'brother-in-law',
                     'sister-in-law', 'husband', 'wife']
me_list = ['i', 'my']


def search_rel_type(sentence):
    for token in word_tokenize(sentence):
        if token.lower() in relationship_list:
            return token.lower()

    return None


def pos_tag_sentence(sentence):
    sentence = re.sub('\W+', ' ', sentence)
    doc = nlp(sentence)

    pos_tagged_sentence = []
    for token in doc:
        pos_tuple = (token.text, token.pos_)
        pos_tagged_sentence.append(pos_tuple)

    #print(pos_tagged_sentence)
    return pos_tagged_sentence


def chunk_sentence(pos_tagged_sentence):
    cp = nltk.RegexpParser(grammar_en)
    result = cp.parse(pos_tagged_sentence)
    #print(f'Chunk: {result}')

    result.draw()
    return result


def extract_rel(sentence):
    me = None

    relative = search_rel_type(sentence)
    if relative:
        chunk_tree = chunk_sentence(pos_tag_sentence(sentence))

        for i, sub_tree in enumerate(chunk_tree):
            if type(sub_tree) is nltk.tree.Tree and sub_tree.label() == 'REL':
                me = sub_tree[0][0][0]

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
positive_text2_en = u'''I've an older brother'''
positive_text3_en = u'''My daughter Lisa lives in Berlin'''
positive_text4_en = u'''My daughter Lisa is moving to London next month.'''

negative_text1_en = u'''her brother'''
negative_text2_en = u'''his father John'''

sentences_pos = [positive_text1, positive_text2, positive_text3, positive_text4, positive_text5]
sentences_neg = [negative_text1, negative_text2, negative_text3, negative_text4, negative_text5, negative_text6]

"""
print('## Positive')
for sentence in sentences_pos:
    print(extract_rel(sentence))

print('\n## Negative')
for sentence in sentences_neg:
    print(extract_rel(sentence))
"""
me, rel = extract_rel(positive_text4_en)
print(me, rel)

