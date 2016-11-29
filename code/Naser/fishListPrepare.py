
from __future__ import print_function
import pickle
import cv2
import os
import numpy as np
import json
from Fish import FishClass


def find_between( s, first, last ):
    try:
        start = s.rfind( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

fishList = []

jsonFile = open('FishDetectionFolder/train/train/ANNOTATION/BET.json', 'r')
valuesBET = json.load(jsonFile)
jsonFile.close()

jsonFile = open('FishDetectionFolder/train/train/ANNOTATION/ALB.json', 'r')
valuesALB = json.load(jsonFile)
jsonFile.close()

jsonFile = open('FishDetectionFolder/train/train/ANNOTATION/DOL.json', 'r')
valuesDOL = json.load(jsonFile)
jsonFile.close()

jsonFile = open('FishDetectionFolder/train/train/ANNOTATION/LAG.json', 'r')
valuesLAG = json.load(jsonFile)
jsonFile.close()

jsonFile = open('FishDetectionFolder/train/train/ANNOTATION/YFT.json', 'r')
valuesYFT = json.load(jsonFile)
jsonFile.close()

jsonFile = open('FishDetectionFolder/train/train/ANNOTATION/SHARK.json', 'r')
valuesSHARK = json.load(jsonFile)
jsonFile.close()

jsonFile = open('FishDetectionFolder/train/train/ANNOTATION/OTHER.json', 'r')
valuesOTHER = json.load(jsonFile)
jsonFile.close()

#######################################################################
print('Adding ALB to the fish list...')

dirAddressOriginal = 'FishDetectionFolder/train/train/ALB_ORIG'
dirAddresses = ['FishDetectionFolder/train/train/ALB/resize', 'FishDetectionFolder/train/train/ALB/resize_col1', 'FishDetectionFolder/train/train/ALB/resize_col2', 'FishDetectionFolder/train/train/ALB/resize_noise']
for dirAddress in dirAddresses:
    for i in range (len(valuesALB)):

        fish = FishClass()
        items = valuesALB[i]
        fish.imageName = find_between(items['filename'], "/", ".")
        fish.fishType = 'ALB'

        if (len(items['annotations'])==0):
            continue

        img = cv2.imread(os.path.join(dirAddress, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fish.fishPixels = np.asarray(gray_image).flatten()


        img = cv2.imread(os.path.join(dirAddressOriginal, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            height, width, channels = img.shape
            fish.original_width = int(width)
            fish.original_heigth = int(height)

        for criteria in items['annotations']:
            if (criteria['class']=='fish'):
                for key, value in criteria.iteritems():
                    fish.fish_X = criteria['x']*(255.0/fish.original_width)
                    fish.fish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.fish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.fish_W = criteria['width']*(255.0/fish.original_width)
            elif (criteria['class']=='non_fish'):
                for key, value in criteria.iteritems():
                    fish.nonfish_X = criteria['x']*(255.0/fish.original_width)
                    fish.nonfish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.nonfish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.nonfish_W = criteria['width']*(255.0/fish.original_width)

            elif (criteria['class'] == 'head'):
                for key, value in criteria.iteritems():
                    fish.head_X = criteria['x']*(255.0/fish.original_width)
                    fish.head_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'tail'):
                for key, value in criteria.iteritems():
                    fish.tail_X = criteria['x']*(255.0/fish.original_width)
                    fish.tail_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'up_fin'):
                for key, value in criteria.iteritems():
                    fish.upfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.upfin_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'low_fin'):
                for key, value in criteria.iteritems():
                    fish.lowfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.lowfin_Y = criteria['y']*(255.0/fish.original_heigth)

        fishList.append(fish)
        print(dirAddress, ':', fish.imageName, ' is set in fish list')

print('ALB added to the fish list!')
####################################################

print('Adding BET to the fish list...')
dirAddressOriginal = 'FishDetectionFolder/train/train/BET_ORIG'
dirAddresses = ['FishDetectionFolder/train/train/BET/resize', 'FishDetectionFolder/train/train/BET/resize_col1', 'FishDetectionFolder/train/train/BET/resize_col2', 'FishDetectionFolder/train/train/BET/resize_noise']
for dirAddress in dirAddresses:
    for i in range (len(valuesBET)):
        fish = FishClass()
        items = valuesBET[i]
        fish.imageName = find_between(items['filename'], "/", ".")
        fish.fishType = 'BET'

        if (len(items['annotations'])==0):
            continue

        img = cv2.imread(os.path.join(dirAddress, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fish.fishPixels = np.asarray(gray_image).flatten()

        img = cv2.imread(os.path.join(dirAddressOriginal, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            height, width, channels = img.shape
            fish.original_width = int(width)
            fish.original_heigth = int(height)


        for criteria in items['annotations']:
            if (criteria['class']=='fish'):
                for key, value in criteria.iteritems():
                    fish.fish_X = criteria['x']*(255.0/fish.original_width)
                    fish.fish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.fish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.fish_W = criteria['width']*(255.0/fish.original_width)
            elif (criteria['class']=='non_fish'):
                for key, value in criteria.iteritems():
                    fish.nonfish_X = criteria['x']*(255.0/fish.original_width)
                    fish.nonfish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.nonfish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.nonfish_W = criteria['width']*(255.0/fish.original_width)

            elif (criteria['class'] == 'head'):
                for key, value in criteria.iteritems():
                    fish.head_X = criteria['x']*(255.0/fish.original_width)
                    fish.head_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'tail'):
                for key, value in criteria.iteritems():
                    fish.tail_X = criteria['x']*(255.0/fish.original_width)
                    fish.tail_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'up_fin'):
                for key, value in criteria.iteritems():
                    fish.upfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.upfin_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'low_fin'):
                for key, value in criteria.iteritems():
                    fish.lowfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.lowfin_Y = criteria['y']*(255.0/fish.original_heigth)

        fishList.append(fish)
        print(dirAddress, ':', fish.imageName, ' is set in fish list')

print('BET added to the fish list!')
####################################################

print('Adding DOL to the fish list...')

dirAddressOriginal = 'FishDetectionFolder/train/train/DOL_ORIG'
dirAddresses = ['FishDetectionFolder/train/train/DOL/resize', 'FishDetectionFolder/train/train/DOL/resize_col1', 'FishDetectionFolder/train/train/DOL/resize_col2', 'FishDetectionFolder/train/train/DOL/resize_noise']
for dirAddress in dirAddresses:
    for i in range (len(valuesDOL)):
        fish = FishClass()
        items = valuesDOL[i]
        fish.imageName = find_between(items['filename'], "/", ".")
        fish.fishType = 'DOL'

        if (len(items['annotations'])==0):
            continue

        img = cv2.imread(os.path.join(dirAddress, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fish.fishPixels = np.asarray(gray_image).flatten()

        img = cv2.imread(os.path.join(dirAddressOriginal, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            height, width, channels = img.shape
            fish.original_width = int(width)
            fish.original_heigth = int(height)


        for criteria in items['annotations']:
            if (criteria['class']=='fish'):
                for key, value in criteria.iteritems():
                    fish.fish_X = criteria['x']*(255.0/fish.original_width)
                    fish.fish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.fish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.fish_W = criteria['width']*(255.0/fish.original_width)
            elif (criteria['class']=='non_fish'):
                for key, value in criteria.iteritems():
                    fish.nonfish_X = criteria['x']*(255.0/fish.original_width)
                    fish.nonfish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.nonfish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.nonfish_W = criteria['width']*(255.0/fish.original_width)

            elif (criteria['class'] == 'head'):
                for key, value in criteria.iteritems():
                    fish.head_X = criteria['x']*(255.0/fish.original_width)
                    fish.head_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'tail'):
                for key, value in criteria.iteritems():
                    fish.tail_X = criteria['x']*(255.0/fish.original_width)
                    fish.tail_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'up_fin'):
                for key, value in criteria.iteritems():
                    fish.upfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.upfin_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'low_fin'):
                for key, value in criteria.iteritems():
                    fish.lowfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.lowfin_Y = criteria['y']*(255.0/fish.original_heigth)

        fishList.append(fish)
        print(dirAddress, ':', fish.imageName, ' is set in fish list')

print('DOL added to the fish list!')
####################################################
print('Adding YFT to the fish list...')

dirAddressOriginal = 'FishDetectionFolder/train/train/YFT_ORIG'
dirAddresses = ['FishDetectionFolder/train/train/YFT/resize', 'FishDetectionFolder/train/train/YFT/resize_col1', 'FishDetectionFolder/train/train/YFT/resize_col2', 'FishDetectionFolder/train/train/YFT/resize_noise']
for dirAddress in dirAddresses:
    for i in range (len(valuesYFT)):
        fish = FishClass()
        items = valuesYFT[i]
        fish.imageName = find_between(items['filename'], "/", ".")
        fish.fishType = 'YFT'

        if (len(items['annotations'])==0):
            continue

        img = cv2.imread(os.path.join(dirAddress, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fish.fishPixels = np.asarray(gray_image).flatten()

        img = cv2.imread(os.path.join(dirAddressOriginal, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            height, width, channels = img.shape
            fish.original_width = int(width)
            fish.original_heigth = int(height)


        for criteria in items['annotations']:
            if (criteria['class']=='fish'):
                for key, value in criteria.iteritems():
                    fish.fish_X = criteria['x']*(255.0/fish.original_width)
                    fish.fish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.fish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.fish_W = criteria['width']*(255.0/fish.original_width)
            elif (criteria['class']=='non_fish'):
                for key, value in criteria.iteritems():
                    fish.nonfish_X = criteria['x']*(255.0/fish.original_width)
                    fish.nonfish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.nonfish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.nonfish_W = criteria['width']*(255.0/fish.original_width)

            elif (criteria['class'] == 'head'):
                for key, value in criteria.iteritems():
                    fish.head_X = criteria['x']*(255.0/fish.original_width)
                    fish.head_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'tail'):
                for key, value in criteria.iteritems():
                    fish.tail_X = criteria['x']*(255.0/fish.original_width)
                    fish.tail_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'up_fin'):
                for key, value in criteria.iteritems():
                    fish.upfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.upfin_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'low_fin'):
                for key, value in criteria.iteritems():
                    fish.lowfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.lowfin_Y = criteria['y']*(255.0/fish.original_heigth)

        fishList.append(fish)
        print(dirAddress, ':', fish.imageName, ' is set in fish list')

print('YFT added to the fish list!')
####################################################
print('Adding OTHER to the fish list...')

dirAddressOriginal = 'FishDetectionFolder/train/train/OTHER_ORIG'
dirAddresses = ['FishDetectionFolder/train/train/OTHER/resize', 'FishDetectionFolder/train/train/OTHER/resize_col1', 'FishDetectionFolder/train/train/OTHER/resize_col2', 'FishDetectionFolder/train/train/OTHER/resize_noise']
for dirAddress in dirAddresses:
    for i in range (len(valuesOTHER)):
        fish = FishClass()
        items = valuesOTHER[i]
        fish.imageName = find_between(items['filename'], "/", ".")
        fish.fishType = 'OTHER'

        if (len(items['annotations'])==0):
            continue

        img = cv2.imread(os.path.join(dirAddress, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fish.fishPixels = np.asarray(gray_image).flatten()

        img = cv2.imread(os.path.join(dirAddressOriginal, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            height, width, channels = img.shape
            fish.original_width = int(width)
            fish.original_heigth = int(height)


        for criteria in items['annotations']:
            if (criteria['class']=='fish'):
                for key, value in criteria.iteritems():
                    fish.fish_X = criteria['x']*(255.0/fish.original_width)
                    fish.fish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.fish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.fish_W = criteria['width']*(255.0/fish.original_width)
            elif (criteria['class']=='non_fish'):
                for key, value in criteria.iteritems():
                    fish.nonfish_X = criteria['x']*(255.0/fish.original_width)
                    fish.nonfish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.nonfish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.nonfish_W = criteria['width']*(255.0/fish.original_width)

            elif (criteria['class'] == 'head'):
                for key, value in criteria.iteritems():
                    fish.head_X = criteria['x']*(255.0/fish.original_width)
                    fish.head_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'tail'):
                for key, value in criteria.iteritems():
                    fish.tail_X = criteria['x']*(255.0/fish.original_width)
                    fish.tail_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'up_fin'):
                for key, value in criteria.iteritems():
                    fish.upfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.upfin_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'low_fin'):
                for key, value in criteria.iteritems():
                    fish.lowfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.lowfin_Y = criteria['y']*(255.0/fish.original_heigth)

        fishList.append(fish)
        print(dirAddress, ':', fish.imageName, ' is set in fish list')

print('OTHER added to the fish list!')
####################################################
print('Adding LAG to the fish list...')

dirAddressOriginal = 'FishDetectionFolder/train/train/LAG_ORIG'
dirAddresses = ['FishDetectionFolder/train/train/LAG/resize', 'FishDetectionFolder/train/train/LAG/resize_col1', 'FishDetectionFolder/train/train/LAG/resize_col2', 'FishDetectionFolder/train/train/LAG/resize_noise']
for dirAddress in dirAddresses:
    for i in range (len(valuesLAG)):
        fish = FishClass()
        items = valuesLAG[i]
        fish.imageName = find_between(items['filename'], "/", ".")
        fish.fishType = 'LAG'

        if (len(items['annotations'])==0):
            continue

        img = cv2.imread(os.path.join(dirAddress, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fish.fishPixels = np.asarray(gray_image).flatten()

        img = cv2.imread(os.path.join(dirAddressOriginal, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            height, width, channels = img.shape
            fish.original_width = int(width)
            fish.original_heigth = int(height)


        for criteria in items['annotations']:
            if (criteria['class']=='fish'):
                for key, value in criteria.iteritems():
                    fish.fish_X = criteria['x']*(255.0/fish.original_width)
                    fish.fish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.fish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.fish_W = criteria['width']*(255.0/fish.original_width)
            elif (criteria['class']=='non_fish'):
                for key, value in criteria.iteritems():
                    fish.nonfish_X = criteria['x']*(255.0/fish.original_width)
                    fish.nonfish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.nonfish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.nonfish_W = criteria['width']*(255.0/fish.original_width)

            elif (criteria['class'] == 'head'):
                for key, value in criteria.iteritems():
                    fish.head_X = criteria['x']*(255.0/fish.original_width)
                    fish.head_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'tail'):
                for key, value in criteria.iteritems():
                    fish.tail_X = criteria['x']*(255.0/fish.original_width)
                    fish.tail_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'up_fin'):
                for key, value in criteria.iteritems():
                    fish.upfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.upfin_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'low_fin'):
                for key, value in criteria.iteritems():
                    fish.lowfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.lowfin_Y = criteria['y']*(255.0/fish.original_heigth)

        fishList.append(fish)
        print(dirAddress, ':', fish.imageName, ' is set in fish list')

print('LAG added to the fish list!')

####################################################
print('Adding SHARK to the fish list...')

dirAddressOriginal = 'FishDetectionFolder/train/train/SHARK_ORIG'
dirAddresses = ['FishDetectionFolder/train/train/SHARK/resize', 'FishDetectionFolder/train/train/SHARK/resize_col1', 'FishDetectionFolder/train/train/SHARK/resize_col2', 'FishDetectionFolder/train/train/SHARK/resize_noise']
for dirAddress in dirAddresses:
    for i in range (len(valuesSHARK)):
        fish = FishClass()
        items = valuesSHARK[i]
        fish.imageName = find_between(items['filename'], "/", ".")
        fish.fishType = 'SHARK'

        if (len(items['annotations'])==0):
            continue

        img = cv2.imread(os.path.join(dirAddress, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fish.fishPixels = np.asarray(gray_image).flatten()

        img = cv2.imread(os.path.join(dirAddressOriginal, fish.imageName+'.jpg'))
        if img is None:
            print('image was None!')
            continue
        else:
            height, width, channels = img.shape
            fish.original_width = int(width)
            fish.original_heigth = int(height)


        for criteria in items['annotations']:
            if (criteria['class']=='fish'):
                for key, value in criteria.iteritems():
                    fish.fish_X = criteria['x']*(255.0/fish.original_width)
                    fish.fish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.fish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.fish_W = criteria['width']*(255.0/fish.original_width)
            elif (criteria['class']=='non_fish'):
                for key, value in criteria.iteritems():
                    fish.nonfish_X = criteria['x']*(255.0/fish.original_width)
                    fish.nonfish_Y = criteria['y']*(255.0/fish.original_heigth)
                    fish.nonfish_H = criteria['height']*(255.0/fish.original_heigth)
                    fish.nonfish_W = criteria['width']*(255.0/fish.original_width)

            elif (criteria['class'] == 'head'):
                for key, value in criteria.iteritems():
                    fish.head_X = criteria['x']*(255.0/fish.original_width)
                    fish.head_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'tail'):
                for key, value in criteria.iteritems():
                    fish.tail_X = criteria['x']*(255.0/fish.original_width)
                    fish.tail_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'up_fin'):
                for key, value in criteria.iteritems():
                    fish.upfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.upfin_Y = criteria['y']*(255.0/fish.original_heigth)

            elif (criteria['class'] == 'low_fin'):
                for key, value in criteria.iteritems():
                    fish.lowfin_X = criteria['x']*(255.0/fish.original_width)
                    fish.lowfin_Y = criteria['y']*(255.0/fish.original_heigth)

        fishList.append(fish)
        print(dirAddress, ':', fish.imageName, ' is set in fish list')

print('SHARK added to the fish list!')

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

print (len(fishList))
save_object(fishList, 'FishDetectionFolder/train/train/ANNOTATION/fishList.pkl')


