""" Extract all conversations from raw book data.
Conversations must be indicated by the conversation
starting (») and ending («) characters """

from nltk.tokenize import sent_tokenize

conv_start = "»"
conv_end = "«"

file_name = 'Winnetou_Band1'
with open('../RelationshipDetection/data/' + file_name + '.txt', 'r', encoding='utf-8') as f:
    raw_book_data = f.read()

# remove blank lines from raw book data
book_data = raw_book_data.replace('\n\n', ' ')

# skip the first the first 'n' lines because they only contain meta info
n = 1
headerlines_count = 0
clean_data = ''
for row in book_data.splitlines():
    if headerlines_count > n:
        clean_data += str(row)
    headerlines_count += 1

out_file = file_name + '_conversations'
with open('../RelationshipDetection/data/' + out_file + '.txt', 'w', encoding='utf-8') as f:
    for sentence in sent_tokenize(clean_data):
        print(sentence)
        if conv_start and conv_end in sentence:
            sentence = sentence.replace(conv_start, '')
            sentence = sentence.replace(conv_end, '')
            f.write(sentence + '\n')
