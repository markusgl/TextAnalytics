import spacy
import os
import nltk
import re
import pprint

from abc import ABC, abstractmethod
from nltk.tag import StanfordNERTagger
from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer, CoreNLPParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import word_tokenize, sent_tokenize
from spacy import displacy
from spacy.symbols import nsubj, VERB

java_path = "C:/Program Files/Java/jdk1.8.0_181/bin/java.exe"
os.environ['JAVAHOME'] = java_path


class SpacyAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('de_core_news_sm')
        self.people = set()
        self.locations = set()
        self.relationships = ['sohn', 'tochter', 'enkel', 'enkelin']

    def extract_entities(self, utterance):
        doc = self.nlp(utterance)
        for ent in doc.ents:
            print("Entity: {}, Label: {}".format(ent, ent.label_))
            if ent.label_ == 'PER' or ent.label_ == 'PERSON':
                self.people.add(ent.text)
            if ent.label_ == 'LOC':
                self.locations.add(ent.text)

    def display_dependencies(self, utterance):
        doc = self.nlp(utterance)
        displacy.serve(doc, style='dep')

    def parse_dependencies(self, utterance):
        stanford_nlp = StanfordAnalyzer()
        self.people, self.locations = stanford_nlp.extract_entities(utterance)
        print("People {}".format(self.people))

        doc = self.nlp(utterance)
        #print("---------- NOUN CHUNKS -------------")
        #for chunk in doc.noun_chunks:
        #    print("TEXT: {}, ROOT.TEXT: {}, ROOT.DEP_: {}, ROOT.HEAD.TEXT: {}".format(
        #        chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text))

        print("\n--------- PARSE TREE --------------")
        for token in doc:
            print("TEXT: {}, DEP: {}, HEAD TEXT: {}, HEAD POS: {}, CHILDREN: {}".format(
                token.text, token.dep_, token.head.text, token.head.pos_,
                [child for child in token.children])
                )
            token_children = [child for child in token.children]

            if token.text in self.people or token.text.lower() in self.relationships:
                for child in token_children:
                    if str(child).lower() in self.relationships or str(child) in self.people:
                        print("* RELATIONSHIP TUPLE: ({}, {})".format(token.text, child))

        """
        print("\n-----------ITERATING LOCAL TREE -------------")

        root = [token for token in doc if token.head == token][0]
        subject = list(root.lefts)[0]
        for descendant in subject.subtree:
            assert subject is descendant or subject.is_ancestor(descendant)
            print(descendant.text, descendant.dep_, descendant.n_lefts,
                  descendant.n_rights,
                  [ancestor.text for ancestor in descendant.ancestors])
        """

    def tag_pos(self, utterance):
        doc = self.nlp(utterance)
        nouns = []
        verbs = []
        for token in doc:
            if token.pos_ == 'VERB' and not token.is_stop:
                verbs.append(token)
            elif token.pos_ == 'NOUN':
                nouns.append(token)

        return nouns, verbs


class StanfordAnalyzer:
    def __init__(self):
        model = 'models/dewac_175m_600.crf.ser.gz'
        #model = 'models/hgc_175m_600.crf.ser.gz'
        self.st_ner = StanfordNERTagger(model,
                                        'models/stanford-ner.jar',
                                        encoding='utf-8')
        # start core nlp server
        #server = CoreNLPServer("models/stanford-corenlp.jar",
        #                       "models/stanford-german-corenlp-models.jar")
        #server.start()
        #self.st_parser = CoreNLPParser()
        #self.dep_parser = CoreNLPDependencyParser

    def extract_entities(self, utterance):
        people = set()
        locations = set()
        tokenized_text = word_tokenize(utterance)
        classified = self.st_ner.tag(tokenized_text)

        for entity in classified:
            entity_text = entity[0]
            entity_label = entity[1]
            #print("Entity: {}, Label: {}".format(entity_text, entity_label))
            if entity_label == 'I-PER':
                people.add(entity_text)
            elif entity_label == 'I-LOC':
                locations.add(entity_text)

        return people, locations

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences

if __name__ == '__main__':
    #analyzer = StanfordAnalyzer()
    #analyzer2 = SpacyAnalyzer()
    utterance1 = u'Ich war heute mit meinem Enkel im Zoo.'
    utterance2 = u'Meine Enkelin Lisa und mein Enkel Lukas fliegen morgen nach London. Sie sind zum ersten Mal in England.'
    utterance3 = u'Mein Sohn hat mich heute angerufen.'
    utterance4 = u'My granddaughter Lisa and my grandson Luke are going to fly to London tomorrow.' \
                 u' They are the first time in England.'
    #analyzer2.display_dependencies(utterance2)
    #per, loc = analyzer.extract_entities(utterance3)
    #print(per)
    #print(loc)

    #analyzer2.display_dependencies(utterance2)

    #sdp = StanfordDependencyParser('model/')
    #result = list(sdp.raw_parse(utterance2))
    #print(result)
    nltk.download('averaged_perceptron_tagger')
    print(ie_preprocess(utterance4))

from nltk.chunk.regexp import RegexpParser