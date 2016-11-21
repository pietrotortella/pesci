import json
from scipy import misc
import os,sys

def control_json_conformity(fname, outfilename):
    """
    This function takes as argument the name of a json file without extension, and produces a new json file with lists of
    images with reported errors.


    Type of reported errors:

    Miss some feature: images in which some feature miss (find which one is up to the user)
    Overlapping squares: fish and non_fish squares overlap
    Bad x coordinates: the feature x has wrong coordinates (out of image shapes)
    :param filename:
    :return:
    """

    # Importing coordinates from json file
    with open(fname) as data_file:
        data = json.load(data_file)

    # Initialization of badlists
    nrtype_badlist=[]
    fish_badlist=[]
    nonfish_badlist=[]
    head_badlist = []
    tail_badlist = []
    upfin_badlist = []
    lowfin_badlist = []
    overlap_badlist = []
    for dict_it in data:
        image = misc.imread(dict_it["filename"])
        s0, s1 = image.shape[0], image.shape[1]

        listoftypes = []
        for d in dict_it["annotations"]:
            listoftypes.append(d['class'])

            # If statements checking if the coordinates are inside the image shapes. If not the filename is added to the
            # relative badlist
            if d["class"] == 'fish':
                h = d["height"]
                w = d["width"]
                x = d["x"]
                y = d["y"]
                # Control on the corner coordinates
                if x<0 or y<0 or x>s1 or y>s0:
                    fish_badlist.append(dict_it["filename"])
                # Control for the square coordinates
                if x+w>s1 or y+h>s0:
                    fish_badlist.append(dict_it["filename"])
                XF = range(int(x),int(x+w+1))
                YF = range(int(y),int(y+h+1))

            if d["class"] == 'non_fish':
                h = d["height"]
                w = d["width"]
                x = d["x"]
                y = d["y"]
                # Control on the corner coordinates
                if x<0 or y<0 or x>s1 or y>s0:
                    nonfish_badlist.append(dict_it["filename"])
                # Control for the square coordinates
                if x+w>s1 or y+h>s0:
                    nonfish_badlist.append(dict_it["filename"])
                XNF = range(int(x), int(x + w + 1))
                YNF = range(int(y), int(y + h + 1))

            # Control for overlapping squares
            if d["class"] == 'head':
                x = d["x"]
                y = d["y"]
                # Control on the corner coordinates
                if x<0 or y<0 or x>s1 or y>s0:
                    head_badlist.append(dict_it["filename"])

            if d["class"] == 'tail':
                x = d["x"]
                y = d["y"]
                # Control on the corner coordinates
                if x<0 or y<0 or x>s1 or y>s0:
                    tail_badlist.append(dict_it["filename"])

            if d["class"] == 'low_fin':
                x = d["x"]
                y = d["y"]
                # Control on the corner coordinates
                if x<0 or y<0 or x>s1 or y>s0:
                    lowfin_badlist.append(dict_it["filename"])

            if d["class"] == 'up_fin':
                x = d["x"]
                y = d["y"]
                # Control on the corner coordinates
                if x<0 or y<0 or x>s1 or y>s0:
                    upfin_badlist.append(dict_it["filename"])

        # Control over type and number of classes in dictionary. If not ok, add the filename to the relative badlist
        if listoftypes.count("fish")!=1 or listoftypes.count("non_fish")!=1 or listoftypes.count("up_fin")!=1 \
                or listoftypes.count("low_fin")!=1 or listoftypes.count("head")!=1 or listoftypes.count("tail")!=1:
            nrtype_badlist.append(dict_it["filename"])

        try:
            if len(list(set(XF) & set(XNF)))>0 and len(list(set(YF) & set(YNF)))>0:
                overlap_badlist.append(dict_it["filename"])
        except NameError,e:
            pass


    # Writing the output on new json file
    # Remove the file if exists
    try:
        os.remove(outfilename)
    except OSError:
        pass

    with open(outfilename, 'w') as f:
        f.write('Miss some feature:')
        f.write('\n')
        json.dump(nrtype_badlist, f, indent=4)
        f.write('\n')

        f.write('Overlapping squares:')
        f.write('\n')
        json.dump(overlap_badlist, f, indent=4)
        f.write('\n')

        f.write('Bad fish coordinates:')
        f.write('\n')
        json.dump(fish_badlist, f, indent=4)
        f.write('\n')

        f.write('Bad non_fish coordinates:')
        f.write('\n')
        json.dump(nonfish_badlist, f, indent=4)
        f.write('\n')

        f.write('Bad head coordinates:')
        f.write('\n')
        json.dump(head_badlist, f, indent=4)
        f.write('\n')

        f.write('Bad tail coordinates:')
        f.write('\n')
        json.dump(tail_badlist, f, indent=4)
        f.write('\n')

        f.write('Bad up_fin coordinates:')
        f.write('\n')
        json.dump(upfin_badlist, f, indent=4)
        f.write('\n')

        f.write('Bad low_fin coordinates:')
        f.write('\n')
        json.dump(lowfin_badlist, f, indent=4)


if __name__ == '__main__':
    filename = sys.argv[1]
    outputfilename = sys.argv[2]
    control_json_conformity(filename,outputfilename)
