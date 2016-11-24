import os,json
import copy

jsonInput = '/home/terminale5/NemoChase_Project/all_jsons/paolo.json'

with open(jsonInput) as data_file:
    data = json.load(data_file)

print len(data)

# Cleaning input dictionary list json
while {} in data: # remove possible empty dictionaries
    data.remove({})

data_copy = copy.copy(data)
for it in data: # remove possible dictionaries with empty 'annotations' value
    if it['annotations'] == []:
        data_copy.remove(it)

print len(data_copy)

with open('/home/terminale5/NemoChase_Project/all_jsons/paolo_correct.json', 'w') as f:
    json.dump(data_copy, f, indent=4)


