""" Semi supervised approach based on Snowball/BREDS"""
import re
import spacy
import numpy as np

from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.tag import StanfordNERTagger
import pandas as pd
from scipy import spatial
model = '../models/dewac_175m_600.crf.ser.gz'
#model = '../models/hgc_175m_600.crf.ser.gz'
#model = '../models/german.conll.germeval2014.hgc_175m_600.crf.ser.gz'

st = StanfordNERTagger(model,
                       '../models/stanford-ner.jar',
                       encoding='utf-8')

nlp = spacy.load('de')
me_list = ['ich', 'meine', 'mein']

training_text = u'''Meine Enkelin Lisa, ihre Freundin Laura und mein Enkel Lukas fliegen morgen nach London. Sie sind zum ersten Mal in England. 
Peter und Maria gehen morgen ins Kino. Ich und mein Sohn gehen heute zum Fußball. Ich gehe mit Johann in den Zoo.
Meine Tochter Christina und mein Sohn Luis fahren nach Berlin. Ich bin geboren zu York im Jahre 1632, als Kind angesehener Leute, die ursprünglich nicht aus jener Gegend stammten. 
Mein Vater, ein Ausländer, aus Bremen gebürtig, hatte sich zuerst in Hull niedergelassen, war dort als Kaufmann zu 
hübschem Vermögen gekommen und dann, nachdem er sein Geschäft aufgegeben hatte, nach York gezogen. 
Hier heiratete er meine Mutter, eine geborene Robinson.
Ich hatte zwei ältere Brüder. Der eine von ihnen, welcher als Oberstleutnant bei einem englischen, 
früher von dem berühmten Oberst Lockhart befehligten Infanterieregiment in Flandern diente, 
fiel in der Schlacht bei Dünkirchen. Was aus dem jüngeren geworden ist, habe ich ebenso wenig in Erfahrung bringen können, 
als meine Eltern je Kenntnisse von meinen eignen Schicksalen erhalten haben.'''

training_dict = {}

for sentence in sent_tokenize(training_text):
    ner_tuples = st.tag(sentence.split())
    rel_tuples = ()

    for ner_tuple in ner_tuples:
        if 'I-PER' in ner_tuple:
            rel_tuples += ('PER',)
        elif ner_tuple[0].lower() in me_list:
            rel_tuples += ('PME',)

    if len(rel_tuples) >= 2:
        clean_sentence = re.sub(r'\W', ' ', sentence)  # remove non-word characters
        clean_sentence = re.sub(r'\s{2,}', ' ', clean_sentence)  # remove two or more consecutive whitespaces
        if rel_tuples in training_dict.keys():
            curr_values = training_dict[rel_tuples]
            curr_values.append(clean_sentence)
        else:
            curr_values = [clean_sentence]

        training_dict[rel_tuples] = curr_values


feature_columns = ['m1', 'm2', 'before_m1', 'after_m2', 'between_words', 'dep_path']
features = pd.DataFrame(columns=feature_columns)

features_list = []  # for TF-IDF vectorization

for sentence in sent_tokenize(training_text):
    sentence = re.sub(r'\W', ' ', sentence)
    sentence = re.sub(r'\s{2,}', ' ', sentence)

    doc = nlp(sentence)
    ner_tuples = st.tag(sentence.split())  # Stanford NER Tagger
    ners = [ner_tuple for ner_tuple in ner_tuples if 'I-PER' in ner_tuple]

    if len(ners) >= 2:
        for i in range(len(ners) - 1):

            # extract both entities
            m1 = ners[i][0]
            m2 = ners[i + 1][0]

            # Dependecy parsing
            dep_path = []
            for chunk in doc.noun_chunks:
                if chunk.root.text == m1 or chunk.root.text == m2:
                    dep_path.append([chunk.root.text, chunk.root.dep_, chunk.root.head.text])

            # find between words
            start_pos_m1 = sentence.find(m1)
            start_pos = start_pos_m1 + len(m1)

            start_pos_m2 = sentence.find(m2)
            end_pos = start_pos_m2

            between = sentence[start_pos + 1:end_pos]
            between_words = []
            for word in word_tokenize(between):
                between_words.append(word)

            beforeM1 = sentence[:start_pos_m1 - 1]
            afterM2 = sentence[start_pos_m2 + len(m2):]

            beforeM1_list = word_tokenize(beforeM1)
            afterM2_list = word_tokenize(afterM2)

            #print(f'm1: {m1}, m2: {m2}, beforeM1: {beforeM1}, afterM2: {afterM2},   between_words: {between_words}, dep_path: {dep_path}')
            data = {'m1': m1, 'm2': m2, 'before_m1': beforeM1_list, 'after_m2': afterM2_list,
                    'between_words': between_words, 'dep_path': dep_path}

            training_ex = pd.Series(data, index=feature_columns)
            features = features.append(training_ex, ignore_index=True)
            context = [beforeM1, between, afterM2]
            features_list.append(context)


#model = KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True, limit=25000)
#print(model.wv['hello'])

# TF-IDF
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
feature_columns = ['BEF', 'BET', 'AFT']
seed_features = pd.DataFrame(columns=feature_columns)
vectors = []
norm_len = 0

for i, row in enumerate(features_list):
    X = vectorizer.fit_transform(row)
    e1 = features.iloc[i]['m1']
    e2 = features.iloc[i]['m2']
    data = {'BEF': X.toarray()[0], 'BET': X.toarray()[1], 'AFT': X.toarray()[2]}
    ex = pd.Series(data, index=feature_columns)
    seed_features = seed_features.append(ex, ignore_index=True)

    if len(X.toarray()[0]) > norm_len:
        norm_len = len(X.toarray()[0])
    if len(X.toarray()[1]) > norm_len:
        norm_len = len(X.toarray()[1])
    if len(X.toarray()[2]) > norm_len:
        norm_len = len(X.toarray()[2])


def compute_similarity(t1, t2):
    bef1 = t1[0]
    bef2 = t2[0]
    bet1 = t1[1]
    bet2 = t2[1]
    aft1 = t1[2]
    aft2 = t2[2]

    bef_sim = 1 - spatial.distance.cosine(bef1, bef2)
    bet_sim = 1 - spatial.distance.cosine(bet1, bet2)
    aft_sim = 1 - spatial.distance.cosine(aft1, aft2)

    sim = bef_sim + bet_sim + aft_sim
    # print(f'Sim({i},{j}) = {sim}')

    return sim


def get_seed_vector(index):
    i = index
    bef = seed_features.iloc[i]['BEF']
    bet = seed_features.iloc[i]['BET']
    aft = seed_features.iloc[i]['AFT']

    # normalize vectors (pad with zeros)
    if len(bef) < norm_len:
        bef = np.pad(bef, (0, norm_len - len(bef)), 'constant')
    if len(bet) < norm_len:
        bet = np.pad(bet, (0, norm_len - len(bet)), 'constant')
    if len(aft) < norm_len:
        aft = np.pad(aft, (0, norm_len - len(aft)), 'constant')

    t = [bef, bet, aft]
    return t


threshold = 0.9
instances = []

for i in range(len(features_list)):
    t = get_seed_vector(i)
    instances.append(t)

# Single pass clustering
clusters = {}
clusters[0] = [instances[0]]
patterns = [clusters[0]]

for i, instance in enumerate(instances):
    for j, cluster in enumerate(patterns):
        if not i == j:
            sim = compute_similarity(instance, cluster[j])

            if sim >= threshold:
                clusters[j].append(instance)
                # for i, c in enumerate(cluster):
                #    cluster = np.add(cluster, instance)
                # cluster.append(instance)

            else:
                # print('h')
                clusters[j + 1] = [instance]
                patterns = [clusters[j + 1]]

