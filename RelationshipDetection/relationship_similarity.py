import nltk
import re
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from gensim.models import Word2Vec, KeyedVectors

# Google word2vec
#model_name = 'models/GoogleNews-vectors-negative300.bin'
#model = KeyedVectors.load_word2vec_format(model_name, binary=True)

# (king - man) + woman
#result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)


training_sentences = []
with open('data/Robinson_Crusoe.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
    sentences = nltk.sent_tokenize(raw_text)

for sentence in sentences:
    clean_sentence = re.sub(r'\W', ' ', sentence)
    clean_sentence = re.sub(r'\s{2,}', ' ', clean_sentence)
    #print(clean_sentence)
    tokenized_sentence = nltk.word_tokenize(clean_sentence)
    training_sentences.append(tokenized_sentence)

model = Word2Vec(training_sentences, sg=1, size=100, window=5, min_count=5, workers=4)

# Plot model

# fit model to 2D
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# scatter plot
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)

for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

# plt.savefig('wordvectors.png')
plt.show()
