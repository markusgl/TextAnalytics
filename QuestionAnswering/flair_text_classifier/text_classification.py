from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class FlairClassifier:
    def __init__(self):
        try:
            self.classifier = TextClassifier.load_from_file(BASE_DIR.joinpath('./models/best-model.pt'))
        except FileNotFoundError as ex:
            print(ex)

    @staticmethod
    def train():
        # load training data in FastText format
        corpus = NLPTaskDataFetcher.load_classification_corpus(Path('./'),
                                                               test_file='./data/test.txt',
                                                               train_file='./data/train.txt')

        # Combine different embeddings:
        # Glove word ebmeddings + Flair contextual string embeddings
        word_embeddings = [WordEmbeddings('glove'),
                           FlairEmbeddings('news-forward-fast'),
                           FlairEmbeddings('news-backward-fast')]
        # use LSTM based method for combining the different embeddings
        document_embeddings = DocumentLSTMEmbeddings(word_embeddings,
                                                     hidden_size=512,
                                                     reproject_words=True,
                                                     reproject_words_dimension=256)

        classifier = TextClassifier(document_embeddings,
                                    label_dictionary=corpus.make_label_dictionary(),
                                    multi_label=False)

        trainer = ModelTrainer(classifier, corpus)
        trainer.train('./models', max_epochs=10)

    def validate(self, question):
        sentece = Sentence(question)
        self.classifier.predict(sentece)

        category = sentece.labels[0].value
        score = sentece.labels[0].score

        return category, score


if __name__ == '__main__':
    tc = FlairClassifier()
    tc.train()
