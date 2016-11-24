import os,sys,json


jsonInput = '/home/terminale5/NemoChase_Project/all_jsons/adri.json'

# Extracting data from json input
with open(jsonInput) as data_file:
    data = json.load(data_file)

for d in data:
    oldclass = os.path.basename(os.path.dirname(d['filename']))
    newclass = 'YFT'
    root = os.path.dirname(os.path.dirname(d['filename']))
    newpath = os.path.join(root,newclass)
    head,tail = os.path.split(d['filename'])
    newfilename = os.path.join(newpath,tail)
    print newfilename
    d['filename'] = newfilename

with open('/home/terminale5/NemoChase_Project/all_jsons/adri_correct.json', 'w') as f:
    json.dump(data, f, indent=4)
