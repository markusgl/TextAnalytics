from gensim.models import KeyedVectors

ger_model = KeyedVectors.load_word2vec_format('../models/german.model', binary=True)


son_vector = ger_model.wv['Sohn']
myson_vector = sum([ger_model.wv['Mein'], ger_model.wv['Sohn']])
print(ger_model.similarity(son_vector, myson_vector))
