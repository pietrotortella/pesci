##IMPORTING REQUIRED PACKAGES

import json

import cv2

import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------

##DEFINING PRELIMINARY FUNCTIONS

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

    return [x,y]

def rectannot(jsn,label):

    ''' takes a json file and a label. It returns the list of dictionaries in the json file corresponding to annotation having class @label).

    @label could be "fish" or "non_fish" '''

    data = json.loads(jsn)

    annotations = []

    for i in range(len(data)):

        for j in range(len(data[i]["annotations"])):

            data[i]["annotations"][j].update({"filename": data[i]["filename"][2:]})

            annotations.append(data[i]["annotations"][j])

    rectangles = [annot for annot in annotations if annot["class"] == label]

    rect_list = []

    for rectangle in rectangles:

        img = cv2.imread(getPath(rectangle))

        rect_img = rect(img, rectangle)

        rect_list.append({getPath(rectangle)[-13:]:rect_img})

    return rect_list

def pointannot(jsn,label):

    ''' takes a json file and a point label. It returns the list of the dictionaries in the json file corresponding to annotation having class @label).

    @label could be "head" or "tail" or "up_fin" or low_fin'''

    data = json.loads(jsn)

    annotations = []

    for i in range(len(data)):

        for j in range(len(data[i]["annotations"])):

            data[i]["annotations"][j].update({"filename": data[i]["filename"][2:]})

            annotations.append(data[i]["annotations"][j])

    points = [annot for annot in annotations if annot["class"] == label]

    point_list = []

    for point in points:

        pc = pnt(point)

        point_list.append({getPath(point)[-13:]:pc})

    return point_list

#--------------------------------------

##CREATING FOR EACH ANNOTATION CLASS (FISH,NON_FISH, HEAD,TAIL,UP_FIN,DOWN_FIN) A LIST OF ALL THE ANNOTATIONS IN THAT CLASS

#folder containing the json file

json_path = "/home/claudia/Desktop/Kaggle_challenge/train/train/annotations/TIN_DOL.json"

#path of the folder containing all the folders of fish classes

mypath = "/home/claudia/Desktop/Kaggle_challenge/train/train"

jsn_file = open(json_path).read()

FISH_list= rectannot(jsn_file,"fish")

NONFISH_list=rectannot(jsn_file,"non_fish")

HEAD_list=pointannot(jsn_file,"head")

TAIL_list=pointannot(jsn_file,"tail")

UFIN_list=pointannot(jsn_file,"up_fin")

LFIN_list=pointannot(jsn_file,"low_fin")

##Uncomment the following lines to show the first fish image of the list @FISH_list

#print rectannot(jsn_file,"fish")[0].keys()

#sub_img=rectannot(jsn_file,"fish")[0]['img_07898.jpg']

#print pointannot(jsn_file,"head")[0].keys()

#cd_point=pointannot(jsn_file,"fish")[0]['img_07898.jpg']

#print cd_point

