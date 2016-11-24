import os
from os import listdir
from os.path import isfile, join

import numpy as np
from scipy import misc
import sys




def changecolour(image_file,a,b,c):

    image = np.array(misc.imread(image_file))


    var_R =  np.var(image[:,:,0])
    var_G =  np.var(image[:,:,1])
    var_B =  np.var(image[:,:,2])

    Red = np.ndarray.flatten(image[:,:,0])
    Green  = np.ndarray.flatten(image[:,:,1])
    Blue = np.ndarray.flatten(image[:,:,2])



    RG = np.cov(Red, Green)
    RB = np.cov(Red, Blue)
    GB = np.cov(Green, Blue)

    cov = np.array([[var_R ,RG[0][1] , RB[0][1]], [RG[1][0] , var_G, GB[0][1]],[RB[1][0] ,GB[1][0] ,var_B]])



    w, v = np.linalg.eig(cov)

    autoval= np.array([[w[0]*a,w[1]*b,w[2]*c]])

    delta_channels = np.dot(v,np.transpose(autoval))

    img_mod=np.zeros(image.shape)
    img_mod[:,:,0] = image[:,:,0]+delta_channels[0]
    img_mod[:, :, 1] = image[:, :, 1] + delta_channels[1]
    img_mod[:, :, 2] = image[:, :, 2] + delta_channels[2]

    return img_mod, image_file


def colors_krizhevsky(mypath):

    newpath = mypath+'_color'

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    os.chdir(mypath) #porta nella cartella al percorso contenuto in mypath

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]   #listdir fa una dir del contenuto della cartella in mypath



    for f in onlyfiles:

        img_mod1, image_file = changecolour(f, np.random.normal(0,0.2), np.random.normal(0,0.2),np.random.normal(0,0.2))
        os.chdir(newpath)
        misc.imsave(image_file[0:-4] + '_col' + str(1) + '.jpg', img_mod1)
        print 'image ' + image_file[0:-4] + '_col' +str(1)+'.jpg created \n'
        os.chdir(mypath)

        img_mod2, image_file = changecolour(f, np.random.normal(0,0.2), np.random.normal(0,0.2),np.random.normal(0,0.2))
        os.chdir(newpath)
        misc.imsave(image_file[0:-4] + '_col' + str(2) + '.jpg', img_mod2)

        print 'image ' + image_file[0:-4] + '_col' + str(2) + '.jpg created \n'
        os.chdir(mypath)


if __name__ == '__main__':
    filename = sys.argv[1]
    colors_krizhevsky(filename)



