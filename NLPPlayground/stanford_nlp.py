import stanfordnlp


#stanfordnlp.download('de')
models_dir = 'C:/Users/marku/develop/Models/stanfordnlp_resources'
lang = 'de'
#text_en = 'Barack Obama was born in Hawaii.  He was elected president in 2008.'
text = 'Albert Einstein wurde in Ulm geboren.'

nlp = stanfordnlp.Pipeline(models_dir=models_dir, lang=lang, mode='predict')
doc = nlp(text)
doc.sentences[0].print_dependencies()
