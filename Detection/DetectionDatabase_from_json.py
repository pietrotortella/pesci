##Importing required packages

import json

import cv2
import math
import numpy as np

import matplotlib.pyplot as plt

#-----------------------------------------
#Getting the target set from the json file

json_path="/home/terminale4/Desktop/kaggle_FishCompetition/train/train/annotations/ALL_clean.json"
#json_path = "/home/terminale4/Desktop/kaggle_FishCompetition/train/train/annotations/TIN_DOL.json"
mypath = "/home/terminale4/Desktop/kaggle_FishCompetition/train/train/"
jsn_file = open(json_path).read()
jsn_data = json.loads(jsn_file)

def getData(jsn_data):
    l = []
    for i in range(3):
        file = mypath+jsn_data[i]["filename"][2:]
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (96, 96))
        img = [item for sublist in img for item in sublist]
        l.append(img)
    return l

data = getData(jsn_data)
print len(data)


# #---------------------
# #Getting the target set from the json file
#
#
def annot_dict(annots,label):
    list = [annot for annot in annots if annot["class"] == label]
    return list[0]
#
#
def getTarget(jsn_data):
    l=[]
    for i in range(len(jsn_data)):
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
#print target


def getFish(jsn_data):
    l = []
    for i in range(len(jsn_data)):
        file = mypath+jsn_data[i]["filename"][2:]
        annots = jsn_data[i]["annotations"]
        x = int(annot_dict(annots, "fish")["x"])
        y = int(annot_dict(annots, "fish")["y"])
        h = int(annot_dict(annots, "fish")["height"])
        w = int(annot_dict(annots, "fish")["width"])
        img = cv2.imread(file)
        img = img[y:y + h, x:x + w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (96, 96))
        img = [item for sublist in img for item in sublist]
        l.append(img)
    return l


data_fish=getFish(jsn_data)
print len(data_fish)


def getAngleTarget(jsn_data):
    thetas=[]
    for i in range(len(jsn_data)):
        annots = jsn_data[i]["annotations"]
        yhead=annot_dict(annots, "head")["y"]
        xhead=annot_dict(annots, "head")["x"]
        ytail=annot_dict(annots, "tail")["y"]
        xtail=annot_dict(annots, "tail")["x"]
        slope = (yhead - ytail) / (xhead - xtail+ 10 **-8)
        theta = math.degrees(math.atan(slope))
        thetas.append(theta)
    return thetas

angle_target = getAngleTarget(jsn_data)
print len(angle_target)







