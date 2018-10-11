import spacy
nlp = spacy.load('en_coref_sm')
doc = nlp(u'My sister has a dog. She loves him.')

print(doc._.has_coref)
print(doc._.coref_clusters)

