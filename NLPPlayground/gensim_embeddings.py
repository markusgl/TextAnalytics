import nltk
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')

model = KeyedVectors.load_word2vec_format('../../Data/word_embeddings/GoogleNews-vectors-negative300.bin', binary=True, limit=25000)
relationship_list = ['father', 'mother', 'dad', 'daddy', 'mom', 'mommy', 'son', 'daughter', 'brother', 'sister',
                     'grandchild', 'grandson', 'granddaughter', 'grandfather', 'grandmother',
                     'grampa', 'grandpa', 'grandma', 'niece', 'nephew', 'uncle', 'aunt', 'cousin', 'brother-in-law',
                     'sister-in-law', 'husband', 'wife']


def get_embeddings_index_from_google_model():
    embeddings_index = {}
    for word in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[word]]
        embeddings_index[model.wv.index2word[word]] = embedding_vector

    return embeddings_index

embedding_index = get_embeddings_index_from_google_model()
son_vector = embedding_index.get('son')
my_vector = embedding_index.get('my')

myson = sum(son_vector, my_vector)

sentence = "his mommy"
words = [word for word in word_tokenize(sentence)]


highest_score = 0
highest_rel = None

for rel in relationship_list:
    try:
        score = model.n_similarity(words, [rel])

        if score > highest_score:
            highest_score = score
            highest_rel = rel
    except KeyError as err:
        print(err)

print(highest_score)
print(highest_rel)

#print(f"son: {model.n_similarity(words, ['son'])}")
#print(f"daughter: {model.n_similarity(words, ['daughter'])}")
#print(f'father: {model.n_similarity(words, ["father"])}')
#print(f'mother: {model.n_similarity(words, ["mother"])}')

