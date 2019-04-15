"""
Take randomly 1000 utterances from persona-chat and 1000 utterances from Friends TV Corpus
"""
from random import randint

output_file = '../data/validation/experimental_val_set_new.txt'


def create_random_line_numbers(length):
    random_lines = []
    for i in range(500):
        random_line_number = randint(0, length)
        while random_line_number in random_lines:  # avoid duplicates
            random_line_number = randint(0, length)
        random_lines.append(random_line_number)

    return random_lines


# persona-chat
lines_to_extract = sorted(create_random_line_numbers(length=131437))
with open('../data/validation/persona-chat_conversations.txt', 'r', encoding='utf-8') as f:
    count = 0
    data = ''
    for line in f.readlines():
        if count in lines_to_extract:
            data += line
        count += 1

with open(output_file, 'a', encoding='utf-8') as f:
    f.write(data)

# Friends TV Corpus
lines_to_extract = sorted(create_random_line_numbers(length=60848))
with open('../data/Friends_TV_Corpus/friends_conversations.txt', 'r', encoding='utf-8') as f:
    count = 0
    data = ''
    for line in f.readlines():
        if count in lines_to_extract:
            data += line
        count += 1

with open(output_file, 'a', encoding='utf-8') as f:
    f.write(data)

