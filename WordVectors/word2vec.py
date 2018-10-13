from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def train_word2vec(sentences):
    # start word embeddings training
    print("start training word2vec...")
    model = Word2Vec(sentences, min_count=1)
    print("training completed")
    print("vocabulary length %i" % len(model.wv.vocab))
    model.save('models/w2vmodel.bin')


# Visualization using matplotlib and PCA
def plot_model(model_file):
    model = Word2Vec.load(model_file)

    # fit model to 2D
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    print(X.shape)

    result = pca.fit_transform(X)
    print(result)

    # scatter plot
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    print(words)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    #plt.savefig('wordvectors.png')
    plt.show()


training_sentences = [['He', 'is', 'not', 'lazy'],
                      ['He', 'is', 'intelligent'],
                      ['He', 'is', 'smart']]
train_word2vec(training_sentences)
plot_model('models/w2vmodel.bin')

# Google word2vec
model_name = 'models/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_name, binary=True)
dog_vector = model['dog']
cat_vector = model['cat']

# (king - man) + woman
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

X = np.vstack((dog_vector, cat_vector))
print(X.shape)
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# scatter plot
plt.scatter(result[:, 0], result[:, 1])
words = ['dog', 'cat']
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
