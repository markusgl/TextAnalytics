import spacy
from spacy import displacy

nlp = spacy.load('en')
#text = u'Barack Obama is married to Michele Obama.'
#text = u'''I have two children. My daughter Lisa lives in Berlin, my son is studying in Munich.'''
#text = u'''Peter is the father of Tom.'''
utterance7 = u'''Peter, Tom's father, will pick us up.'''
#text = u'''My daughter Lisa is moving to London next month.'''
#text = u'''Anna and her brother Max are going to school.'''
utterance8 = u'''i've a 9 year old son as well .'''
utterance9 = u'''I have a son, he is 16 years old, and my dad, he is retired now.'''
utterance10 = u"So uh, Monica is Ross's sister."
utterance11 = u'Protesters seized several pumping stations, holding 127 Shell workers hostage.'
utterance12 = u"So Monica is Ross's sister, right?"
utterance13 = u'Protesters seized several pumping stations.'
utterance14 = u'Troops recently have raided churches.'
utterance16 = u'My daughter Lisa is moving to London next month.'

utterance15 = u'Peter is the father of Tom.'
utterance17 = u'''Tom's sister Lisa lives in London now.'''
utterance18 = u'''Peter, Tom's father, is a lawyer.'''
utterance19 = u"Monica is Ross's sister, right?"
utterance20 = u'''He's Angela's... brother.'''
utterance21 = u'''I prefer the morning flight through Denver'''

text = utterance21
doc = nlp(text)
displacy.serve(doc, style='dep', options={'compact': False})
