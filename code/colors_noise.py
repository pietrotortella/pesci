import os
from scipy import ndimage, misc
from os import listdir
from os.path import isfile, join
import numpy as np
import sys


#add salt and pepper noise to the image
#p is the probability of having salt noise and pepper noise (equal probability of having salt or pepper noise)


def saltpepper(image_file,p):

    if isinstance(image_file, str):
        image = np.array(misc.imread(image_file))
    else:
        image = image_file
    output = image.copy()
    lim = 1-p
    rdn = np.random.random(image.shape)  # it creates a matrix of random floats in the interval [0,1)
    A = rdn < p
    B = rdn > lim
    output[A]=0
    output[B]=255
    return output, image_file


# function that produces a new image with randomly modified colors
# var: a,b,c random floats in the interval [0. 1)
def changecolour(image_file,a,b,c):
    image = np.array(misc.imread(image_file))
    img_mod = [image[:,:,0]*a,image[:,:,1]*b,image[:,:,2]*c]
    img_mod = np.swapaxes(img_mod, 0, 1)
    img_mod = np.swapaxes(img_mod, 1, 2)
    return img_mod, image_file


def Colors_noise(mypath):
    # mypath is the path that contains the images
    # newpath is the path in which the new images will be stored
    newpath = mypath+'_color'

    # if the path already exists the software doesn't do anything, if it creates a new folder
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # change the current directory to mypath
    os.chdir(mypath)

    #listdir returns a list with the content of the directory specified in mypath

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    #it cycles on every image inside the folder specified in mypath and save the modified images in the folder newpath
    for f in onlyfiles:
        img_mod1, image_file = changecolour(f, np.random.random([1]), np.random.random([1]),np.random.random([1]))
        os.chdir(newpath)
        misc.imsave(image_file[0:-4] + '_col' + str(1) + '.jpg', img_mod1)

        #print 'image ' + image_file[0:-4] + '_col' + str(1) + '.jpg created \n'
        os.chdir(mypath)

        img_mod2, image_file = changecolour(f, np.random.random([1]),np.random.random([1]),np.random.random([1]) )
        os.chdir(newpath)
        misc.imsave(image_file[0:-4] + '_col' + str(2) + '.jpg', img_mod2)

        #print 'image ' + image_file[0:-4] + '_col' + str(2) + '.jpg created \n'
        os.chdir(mypath)

        img_mod3, image_file = saltpepper(f, np.random.random([1]) * 0.05)
        os.chdir(newpath)
        misc.imsave(image_file[0:-4] + '_noise' + '.jpg', img_mod3)
        #print 'image ' + image_file[0:-4] + '_noise' + '.jpg created \n'
        os.chdir(mypath)

    print('OK!')


if __name__ == '__main__':
    filename = sys.argv[1]
    Colors_noise(filename)




