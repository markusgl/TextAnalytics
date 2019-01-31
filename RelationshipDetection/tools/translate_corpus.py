import os
import re

from google.cloud import translate

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:\\Users\\marku\\Documents\\knowledgegraph-219012-adf54eaeb25f.json"
client = translate.Client()


def translate_text(text, target_lang='de'):
    translation = client.translate(text, target_language=target_lang)
    translated_text = translation['translatedText']

    return translated_text


translated_corpus = ''
with open('../data/validation/validation_friends_and_convai.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        clean_line = line.replace('"', '')
        clean_line = re.sub('\s{2,}', ' ', clean_line)
        #translated_corpus += translate_text(clean_line) + '\n'
        translated_corpus += clean_line


with open('../data/validation/validation_friends_and_convai_clean.txt', 'a', encoding='utf-8') as f:
    f.write(translated_corpus)

