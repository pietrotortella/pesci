##IMPORTING REQUIRED PACKAGES
import json
from scipy import misc
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------
##DEFINING PRELIMINARY FUNCTIONS



def rect(image,annot):
    ''' returns the subimage of @image correspoding to the rectangle annotation @annot'''
    x = int(annot["x"])
    h = int(annot["height"])
    y = int(annot["y"])
    w = int(annot["width"])
    return image[y:y + h, x:x + w]

def pnt(annot):
    ''' returns the coordinates of the point annotation @annot'''
    x = int(annot["x"])
    y = int(annot["y"])
    return [y,x]


def rectannot(jsn,label):
    ''' takes a json file and a label. It returns the list of dictionaries in the json file corresponding to annotation having class @label).
    @label could be "fish" or "non_fish" '''

    data = json.loads(jsn)  # store the list of dictionaries contained in the json file

    filenames = [data[i]["filename"] if "/home" in data[i]["filename"]  else mypath + data[i]["filename"][2:] for i in
                 range(len(data))]  # list of paths of all images contained in the json file

    rect_list = []

    for k in range(len(data)):

        for i in range(len(data[k]["annotations"])):
            if data[k]["annotations"][i]["class"] == label:
                fish_dict = data[k]["annotations"][i]
                img = misc.imread(filenames[k])
                rect_img = rect(img, fish_dict)
                rect_list.append({filenames[k][-13:]: rect_img})

    return rect_list



def pointannot(jsn,label):
    ''' takes a json file and a point label. It returns the list of the dictionaries in the json file corresponding to annotation having class @label).
    @label could be "head" or "tail" or "up_fin" or low_fin'''

    data = json.loads(jsn)  # store the list of dictionaries contained in the json file

    filenames = [data[i]["filename"] if "/home" in data[i]["filename"]  else mypath + data[i]["filename"][2:] for i in
                 range(len(data))]  # list of paths of all images contained in the json file

    point_list = []

    for k in range(len(data)):

        for i in range(len(data[k]["annotations"])):
            if data[k]["annotations"][i]["class"] == label:
                point_dict = data[k]["annotations"][i]
                pc = pnt(point_dict)
                point_list.append({filenames[k][-13:]: pc})

    return point_list


#--------------------------------------
##CREATING FOR EACH ANNOTATION CLASS (FISH,NON_FISH, HEAD,TAIL,UP_FIN,DOWN_FIN) A LIST OF ALL THE ANNOTATIONS IN THAT CLASS

#folder containing the json file
json_path = "/home/terminale1/Desktop/Kaggle_challenge/train/annotations/check.json"

#path of the folder containing all the folders of fish classes
mypath = "/home/terminale1/Desktop/Kaggle_challenge/train/"

jsn_file = open(json_path).read()       #read json file



FISH_list= rectannot(jsn_file,"fish")
NONFISH_list=rectannot(jsn_file,"non_fish")
HEAD_list=pointannot(jsn_file,"head")
TAIL_list=pointannot(jsn_file,"tail")
UFIN_list=pointannot(jsn_file,"up_fin")
LFIN_list=pointannot(jsn_file,"low_fin")

print FISH_list,NONFISH_list,HEAD_list,TAIL_list,UFIN_list,LFIN_list


