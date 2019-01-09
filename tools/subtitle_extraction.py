import re

re_timestamp = r'\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}'
re_html = r'<\/?\w*>|\([ÄÜÖA-Z\s]*\)'
re_multline = r'\s*'

file_name = 'TBBT_S10E16_German_subtitle'

with open('../RelationshipDetection/data/' + file_name + '.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    data = f.read()
    data = re.sub(re_timestamp, '', data)
    data = re.sub(re_html, '', data)

out_file = file_name + '_clean'
with open('../RelationshipDetection/data/'+out_file+'.txt', 'w', encoding='utf-8') as wf:
    wf.write(data)
