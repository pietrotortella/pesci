##Importing required packages

import json

import cv2

import numpy as np

import matplotlib.pyplot as plt

#-----------------------------------------
#Getting the target set from the json file


json_path = "/annotations/TIN_DOL.json"
mypath = "/home/claudia/Kaggle_challenge/train/train/"
jsn_file = open(json_path).read()
jsn_data = json.loads(jsn_file)

def getData(jsn_data):
    l = []
    for i in range(3):
        file = mypath+jsn_data[i]["filename"][2:]
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (4, 4))
        img = [item for sublist in img for item in sublist]
        l.append(img)
    return l

data = getData(jsn_data)
print data


#---------------------
#Getting the target set from the json file


def annot_dict(annots,label):
    list = [annot for annot in annots if annot["class"] == label]
    return list[0]


def getTarget(jsn_data):
    l=[]
    for i in range(3):
        a = []
        annots = jsn_data[i]["annotations"]
        a.append(int(annot_dict(annots,"fish")["x"]))
        a.append(int(annot_dict(annots, "fish")["y"]))
        a.append(int(annot_dict(annots, "fish")["height"]))
        a.append(int(annot_dict(annots, "fish")["width"]))
        a.append(int(annot_dict(annots, "non_fish")["x"]))
        a.append(int(annot_dict(annots, "non_fish")["y"]))
        a.append(int(annot_dict(annots, "non_fish")["height"]))
        a.append(int(annot_dict(annots, "non_fish")["width"]))
        a.append(int(annot_dict(annots, "head")["x"]))
        a.append(int(annot_dict(annots, "head")["y"]))
        a.append(int(annot_dict(annots, "tail")["x"]))
        a.append(int(annot_dict(annots, "tail")["y"]))
        a.append(int(annot_dict(annots, "up_fin")["x"]))
        a.append(int(annot_dict(annots, "up_fin")["y"]))
        a.append(int(annot_dict(annots, "low_fin")["x"]))
        a.append(int(annot_dict(annots, "low_fin")["y"]))
        l.append(a)
    return l


target=getTarget(jsn_data)
print target

