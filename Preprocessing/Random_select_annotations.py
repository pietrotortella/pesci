import json,os,sys,random
from collections import Counter


def random_select_annotations(jsonInput,percentage,jsonOutput):

    # Extracting data from json input
    with open(jsonInput) as data_file:
        data = json.load(data_file)

    # List of [index,label] elements
    label_list = []
    for index, dict_it in enumerate(data):
        label_list.append([index, os.path.basename(os.path.dirname(dict_it['filename']))])

    # List of all label occurrences
    all_labels = [x[1] for x in label_list]

    # Dictionary with number of occurrences for each label
    howmany = Counter(all_labels)

    # List of labels
    labels = list(set(i for i in all_labels))

    # Creation of the new list
    newlist = []
    for l in labels:
        n = int(float(percentage) * howmany[l])
        print 'Selecting', n, 'elements of', l, 'type'
        # Extract n dictionaries with file of type l
        indices = [x[0] for x in label_list if x[1] == l]
        selectedindices = random.sample(indices, n)
        for i in selectedindices:
            newlist.append(data[i])

    print 'Number of total inputs selected = ', len(newlist)

    with open(jsonOutput,'w') as f:
        json.dump(newlist,f,indent=4)

#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    jsoninput = sys.argv[1]
    p = sys.argv[2]
    jsonoutput = sys.argv[3]
    random_select_annotations(jsoninput,p,jsonoutput)

