#-----------------------------
##PROGRAM DESCRIPTION

##The program defines two main functions rectannot and pointannot.

#INPUT
# Both the functions take as input a json file containing annotations of some images and a @label.
#For rectannot function the @label should be the annotation name of a rectangular annotation (e.g. "fish", "non_fish")
#For pointannot function the @label should be the annotation name of a point annotation (e.g. "head", "tail", "up_fin", "low_fin")

#OUTPUT
#For both the functions the output is a dictionary having as keys the images filenames and as values the corresponding @label annotations
#(e.g "fish"-annotations, or "head"-annotations)

#----------------------------
##IMPORTING REQUIRED PACKAGES

import json

import cv2

import matplotlib.pyplot as plt

#------------------------------
##DEFINING SOME PRELIMINARY FUNCTIONS

def getPath(annot):

    '''returns the path of the image corresponding to the annotation @annot'''

    return (mypath+annot["filename"])

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

#--------------------------------------------
##DEFINING THE TWO MAIN FUNCTIONS

def rectannot(jsn,label):

    ''' takes a json file and a label. It returns a dictionary corresponding to label annotations).

    label could be "fish" or "non_fish" '''

    data = json.loads(jsn)

    annotations = []

    for i in range(len(data)):

        for j in range(len(data[i]["annotations"])):

            data[i]["annotations"][j].update({"filename": data[i]["filename"][2:]})

            annotations.append(data[i]["annotations"][j])

    rectangles = [annot for annot in annotations if annot["class"] == label]

    rect_dict = {}

    for rectangle in rectangles:

        img = cv2.imread(getPath(rectangle))

        rect_img = rect(img, rectangle)

        rect_dict.update({getPath(rectangle)[-13:]:rect_img})

    return rect_dict


def pointannot(jsn,label):

    ''' takes a json file and a point label. It returns the dictionary corresponding to label annotations).

    label could be "head", "tail", "up_fin" or "low_fin". '''

    data = json.loads(jsn)

    annotations = []

    for i in range(len(data)):

        for j in range(len(data[i]["annotations"])):

            data[i]["annotations"][j].update({"filename": data[i]["filename"][2:]})

            annotations.append(data[i]["annotations"][j])

    points = [annot for annot in annotations if annot["class"] == label]

    point_dict = {}

    for point in points:

        pc = pnt(point)

        point_dict.update({getPath(point)[-13:]:pc})

    return point_dict

#--------------------------------------
## MAIN SCRIPT

#Reading the path where I have my image folder (e.g. the path of the folder DOL)

mypath="/home/claudia/Kaggle_challenge/train/train"

#Opening the json file you want to work on

json_path = "/home/claudia/Kaggle_challenge/github_pesce/pesci/annotations/TIN_DOL.json"

'''folder containing the json file'''

jsn_file = open(json_path).read()

# for each annotation label ("fish", "non_fish", "head", "tail", "fin_up","fin_low") a dictionary having as keys the image filenames and as values
# the annotations with that label is created

FISH_dict= rectannot(jsn_file,"fish")

NONFISH_dict=rectannot(jsn_file,"non_fish")

HEAD_dict=pointannot(jsn_file,"head")

TAIL_dict=pointannot(jsn_file,"tail")

UFIN_dict=pointannot(jsn_file,"up_fin")

LFIN_dict=pointannot(jsn_file,"low_fin")

#---------------------------

## TEST
# for knowing if there are some images having less or more than six annotations

#data = json.loads(jsn_file)
# for i in range(len(data)):
#     if len(data[i]["annotations"])!= 6:
#         print data[i]["filename"][2:]


