import re

re_timestamp = r'\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}'
re_html = r'<\/?\w*>|\([ÄÜÖA-Z\s]*\)'
re_emptyline = r'\n\s*\n'
re_numbers = r'^[0-9]+'

file_name = 'TBBT_S01E02'

data = ''
with open('../RelationshipDetection/data/' + file_name + '.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if not re.match(re_timestamp, line) and not re.match(re_html, line) and not re.match(re_numbers, line) and not line.isspace():
            print(line.replace('- ', ''))
            data += line.replace('- ', '')

out_file = file_name + '_clean'
with open('../RelationshipDetection/data/'+out_file+'.txt', 'w', encoding='utf-8') as wf:
    wf.write(data)


