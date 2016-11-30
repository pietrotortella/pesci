import json
from scipy import misc, io
import os,sys
import numpy as np
import pandas as pd
import copy

def convertToGreyscale(namejpg):
    image_rgb = misc.imread(namejpg)
    image = ((0.2125 * image_rgb[:, :, 0] + 0.7154 * image_rgb[:, :, 1] + 0.0721 * image_rgb[:, :, 2]).astype(float)) / 3
    return image

#-----------------------------------------------------------------------------------------------------------------------

def resize(image):
    s0, s1 = image.shape[0], image.shape[1]
    newimage = misc.imresize(image, (256, 256), interp='bicubic')
    return newimage, s0, s1

#-----------------------------------------------------------------------------------------------------------------------

def resizeDict(dict,X_factor,Y_factor):
    """
    This function calculates, for a given dictionary referring to a specific annotation, its rescaled coordinates, with
    respect to given scaling factors. Two types of calculations are made, depending on which class between "rect" and
    "point" is met.
    :param dict: dictionary of a given annotation
    :param X_factor: horizontal scaling factor
    :param Y_factor: vertical scaling factor
    :return: newdict: dictionary with rescaled coordinates
             newdictTarget: array of coordinates of the newdict dictionary
    """
    rectDict = {'fish', 'non_fish'}
    pointDict = {'head','tail','up_fin','low_fin'}
    if dict["class"] in rectDict:
        h_new = dict["height"]*Y_factor
        w_new = dict["width"]*X_factor
        x_new = dict["x"]*X_factor
        y_new = dict["y"]*Y_factor
        newdict = {
            "type": "rect",
            "class": dict["class"],
            "height": h_new,
            "width": w_new,
            "x": x_new,
            "y": y_new
        }
        newdictTarget = np.array([[x_new, y_new, h_new, w_new]])
    if dict["class"] in pointDict:
        x_new = dict["x"]*X_factor
        y_new = dict["y"]*Y_factor
        newdict = {
            "type": "point",
            "class": dict["class"],
            "x": x_new,
            "y": y_new
        }
        newdictTarget = np.array([[x_new, y_new]])
    return newdict, newdictTarget

#-----------------------------------------------------------------------------------------------------------------------

def resizeJson(dict_in,Xf,Yf):
    """
    This function calculates, for a given dictionary, the coordinates of its annotations after the resizing, with
     respect to given scaling factors. For each calculated dictionary a related dictionary is formed with class
     annotations as keys, and arrays of their coordinate as corresponding values (target values for regression)
    :param dict_in: input dictionary
    :param Xf: horizontal scaling factor
    :param Yf: vertical scaling factor
    :return: dict_out: dictionary with rescaled coordinates annotations
             newdictTargets: dictionary of target values
    """
    dictlist = []
    newdictTargets = {}
    for d in dict_in["annotations"]:
        newdict, newdictTarget = resizeDict(d, Xf, Yf)
        dictlist.append(newdict)
        newdictTargets.update({d["class"]: newdictTarget})

    dict_out = {
        "annotations": dictlist,
        "class": "image",
        "filename": dict_in["filename"]
    }

    return dict_out, newdictTargets

#-----------------------------------------------------------------------------------------------------------------------

def mirroring(dict,s0,s1):
    rectDict = {'fish', 'non_fish'}
    pointDict = {'head','tail','up_fin','low_fin'}
    if dict["class"] in rectDict:
        h_src = dict["height"]
        w_src = dict["width"]
        x_src = dict["x"]
        y_src = dict["y"]

        # Horizontal mirroring
        x_hm = x_src
        y_hm = s0 - 1 - y_src - h_src
        newdict_hm = {
            "type": "rect",
            "class": dict["class"],
            "height": h_src,
            "width": w_src,
            "x": x_hm,
            "y": y_hm
        }

        # Vertical mirroring (along vertical axes)
        x_vm = s1 - 1 - x_src - w_src
        y_vm = y_src
        newdict_vm = {
            "type": "rect",
            "class": dict["class"],
            "height": h_src,
            "width": w_src,
            "x": x_vm,
            "y": y_vm
        }

        # Composition of both mirrorings
        x_hvm = s1 - 1 - x_src - w_src
        y_hvm = s0 - 1 - y_src - h_src
        newdict_hvm = {
            "type": "rect",
            "class": dict["class"],
            "height": h_src,
            "width": w_src,
            "x": x_hvm,
            "y": y_hvm
        }
        newdictTarget_vm = np.array([[x_vm, y_vm, h_src, h_src]])
        newdictTarget_hvm = np.array([[x_hvm, y_hvm, h_src, h_src]])
        newdictTarget_hm = np.array([[x_hm, y_hm, h_src, h_src]])

        return newdict_vm, newdict_hvm, newdict_hm, newdictTarget_vm, newdictTarget_hvm, newdictTarget_hm

    elif dict["class"] in pointDict:
        x_src = dict["x"]
        y_src = dict["y"]

        # Horizontal mirroring
        x_hm = x_src
        y_hm = s0 - 1 - y_src
        newdict_hm = {
            "type": "point",
            "class": dict["class"],
            "x": x_hm,
            "y": y_hm
        }

        # Vertical mirroring
        x_vm = s1 - 1 - x_src
        y_vm = y_src
        newdict_vm = {
            "type": "point",
            "class": dict["class"],
            "x": x_vm,
            "y": y_vm
        }

        # Composition mirroring
        x_hvm = s1 - 1 - x_src
        y_hvm = s0 - 1 - y_src
        newdict_hvm = {
            "type": "point",
            "class": dict["class"],
            "x": x_hvm,
            "y": y_hvm
        }
        newdictTarget_vm = np.array([[x_vm, y_vm]])
        newdictTarget_hvm = np.array([[x_hvm, y_hvm]])
        newdictTarget_hm = np.array([[x_hm, y_hm]])

        return newdict_vm, newdict_hvm, newdict_hm, newdictTarget_vm, newdictTarget_hvm, newdictTarget_hm

#-----------------------------------------------------------------------------------------------------------------------

def mirroringDict(dictionary,s0,s1):
    # Lists containg all the transformed dictionaries of a given image
    dictlist_vm = []
    dictlist_hm = []
    dictlist_hvm = []
    newdictTargets_vm = {}
    newdictTargets_hm = {}
    newdictTargets_hvm = {}
    for d in dictionary["annotations"]:
        newdict_vm, newdict_hvm, newdict_hm, newdictTarget_vm, newdictTarget_hvm, newdictTarget_hm = mirroring(d, s0, s1)
        dictlist_vm.append(
            newdict_vm)  # lists containing each one the mirrored (one mirror only) dictionaries of a given image
        dictlist_hm.append(newdict_hm)
        dictlist_hvm.append(newdict_hvm)
        newdictTargets_vm.update({d["class"]: newdictTarget_vm})
        newdictTargets_hm.update({d["class"]: newdictTarget_hm})
        newdictTargets_hvm.update({d["class"]: newdictTarget_hvm})

    dictfile_vm = {
        "annotations": dictlist_vm,
        "class": "image",
        "filename": dictionary["filename"]
    }

    dictfile_hm = {
        "annotations": dictlist_hm,
        "class": "image",
        "filename": dictionary["filename"]
    }

    dictfile_hvm = {
        "annotations": dictlist_hvm,
        "class": "image",
        "filename": dictionary["filename"]
    }

    return dictfile_vm, dictfile_hvm, dictfile_hm, newdictTargets_vm, newdictTargets_hvm, newdictTargets_hm

#-----------------------------------------------------------------------------------------------------------------------
# Add salt and pepper noise to the image
# p is the probability of having salt noise and pepper noise (equal probability of having salt or pepper noise)
def saltpepper(image, p):
    #image = np.array(misc.imread(image_file))
    output = image.copy()
    lim = 1-p
    rdn = np.random.random(image.shape)  # it creates a matrix of random floats in the interval [0,1)
    A = rdn < p
    B = rdn > lim
    output[A]=0
    output[B]=255
    return output

#-----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION

def buildInputmatrix(jsonInput,imagesPath=None):

    # Extracting data from input json file
    with open(jsonInput) as data_file:
        data = json.load(data_file)

    # Cleaning input dictionary list json
    data_copy = copy.copy(data)
    while {} in data: # remove possible empty dictionaries
        data_copy.remove({})
    data = copy.copy(data_copy)

    data_copy = copy.copy(data)
    for it in data: # remove possible dictionaries with empty 'annotations' value
        if it['annotations'] == []:
            data_copy.remove(it)
    data = copy.copy(data_copy)

    # Initialize input matrix X and target matrix Y
    X = np.empty((0,256*256))
    Y = np.empty((16,0))
    # Initialize nrsample x 1 label column L
    labels = {'ALB':0,'BET':1,'DOL':2,'LAG':3,'SHARK':4,'YFT':5,'OTHER':6,'NoF':7}
    L = np.empty((0,1))
    #L = {}
    # Initialize filename column
    F = np.empty((0,1))

    for dict_it in data:
        # Extracting name of file with its path
        namejpg = dict_it["filename"]

        # Extacting labels
        labelkey = os.path.basename(os.path.dirname(namejpg))
        k = 0
        while k<8:
            L = np.vstack([L,labels[labelkey]])
            k+=1

        # Extracting file name
        fname = os.path.basename(namejpg)

        if imagesPath != None:
            # If imagesPath is not set to None, then each image is searched through the given path
            namejpg = os.path.join(imagesPath, os.path.basename(namejpg))

        # Greyscale image
        image_grey = convertToGreyscale(namejpg)

        # Resize to 256X256
        image_resize, s0, s1 = resize(image_grey) # the original shapes of the image are retain are kept for later
                                                  # calculation of the rescaling factors

        # Resize of dictionary coordinates
        X_factor = 256./s1 # rescaling factor along x axes
        Y_factor = 256./s0 # rescaling factor along y axes
        s0 = 256. # updating dimension (used by mirroring transformations)
        s1 = 256.
        dict_resize, targets_dict = resizeJson(dict_it, X_factor, Y_factor) # input and target dictionary coordinates
        # vertical concatenation of array columns
        targets_column_resize = np.vstack([targets_dict["fish"].T,\
                                    targets_dict["head"].T,\
                                    targets_dict["tail"].T,\
                                    targets_dict["up_fin"].T,\
                                    targets_dict["low_fin"].T,\
                                    targets_dict["non_fish"].T])

        # Mirroring of image
        im_mirV = image_resize[:, ::-1] # vertical axes
        im_mirVH = im_mirV[::-1, :] # vertical + horizonatal axes
        im_mirH = im_mirVH[:, ::-1] # horizontal axes

        # Mirroring of dictionaries
        dictfile_vm, dictfile_hvm, dictfile_hm, newdictTargets_vm,\
        newdictTargets_hvm, newdictTargets_hm = mirroringDict(dict_resize, s0, s1)

        targets_column_vm = np.vstack([newdictTargets_vm["fish"].T,\
                                    newdictTargets_vm["head"].T,\
                                    newdictTargets_vm["tail"].T, \
                                    newdictTargets_vm["up_fin"].T,\
                                    newdictTargets_vm["low_fin"].T,\
                                    newdictTargets_vm["non_fish"].T])

        targets_column_hvm = np.vstack([newdictTargets_hvm["fish"].T,\
                                    newdictTargets_hvm["head"].T,\
                                    newdictTargets_hvm["tail"].T, \
                                    newdictTargets_hvm["up_fin"].T,\
                                    newdictTargets_hvm["low_fin"].T,\
                                    newdictTargets_hvm["non_fish"].T])

        targets_column_hm = np.vstack([newdictTargets_hm["fish"].T,\
                                    newdictTargets_hm["head"].T,\
                                    newdictTargets_hm["tail"].T, \
                                    newdictTargets_hm["up_fin"].T,\
                                    newdictTargets_hm["low_fin"].T,\
                                    newdictTargets_hm["non_fish"].T])

        # Noise of the images
        p = 0.025
        image_noise = saltpepper(image_resize, p)
        image_vm_noise = saltpepper(im_mirV, p)
        image_hvm_noise = saltpepper(im_mirVH, p)
        image_hm_noise = saltpepper(im_mirH, p)

        # Noise of the dictionary coordinates
        targets_column_noise = targets_column_resize
        targets_column_vm_noise = targets_column_vm
        targets_column_hvm_noise = targets_column_hvm
        targets_column_hm_noise = targets_column_hm

        # Input matrix
        # Sort of images: 1) original_resize, 2) mirV, 3) mirVH, 4) mirH, 5) noise, 6) noiseV, 7) noiseVH, 8) noiseH
        Xtemp = np.array([image_resize.flatten(),\
                          im_mirV.flatten(),\
                          im_mirVH.flatten(),\
                          im_mirH.flatten(),\
                          image_noise.flatten(), \
                          image_vm_noise.flatten(), \
                          image_hvm_noise.flatten(), \
                          image_hm_noise.flatten()])
        X = np.vstack([X,Xtemp])

        # Target matrix
        # Y = nrsamples x 16
        Y = np.hstack([Y,\
                       targets_column_resize,\
                       targets_column_vm,\
                       targets_column_hvm,\
                       targets_column_hm,\
                       targets_column_noise,\
                       targets_column_vm_noise,\
                       targets_column_hvm_noise,\
                       targets_column_hm_noise])

        # F = nroftransfrmations+1 x 1 filnames column
        fname_vm = fname[:-4]+'_vm.jpg'
        fname_hvm = fname[:-4]+'_hvm.jpg'
        fname_hm = fname[:-4]+'_hm.jpg'
        fname_sp = fname[:-4]+'_sp.jpg'
        fname_sp_vm = fname[:-4]+'_sp_vm.jpg'
        fname_sp_hvm = fname[:-4]+'_sp_hvm.jpg'
        fname_sp_hm = fname[:-4]+'_sp_hm.jpg'
        F = np.vstack([F,fname,fname_vm,fname_hvm,fname_hm,fname_sp,fname_sp_vm,fname_sp_hvm,fname_sp_hm])

        print 'Image', fname, 'elaborated successfully', ' -->  nrrows:', len(F), '\n'

    Y = np.array(Y.T)/256

    # The following two commands to be uncommented only if we want separate matrices as outputs (otherwise
    # np.hstack doesn't work). L and F are column arrays where each row contains a 1-dim array with single element. Thus
    # their shape is (80,1) and accessing the elements requires two indices. The commands below reshape the matrices to
    # (80,), and on index only is needed.
    # L = L[:,0]
    # F = F[:,0]

    # return np.hstack([F, X, Y, L])
    return F,X,Y,L

def saveCSV(F,X,Y,L,suffix):
    headerF = "FILENAME"
    np.savetxt(suffix+'_F.csv', F, delimiter=",",header=headerF,fmt="%s")

    headerX = ""
    for i in range(256**2):
        headerX+='X_'+str(i)+','
    np.savetxt(suffix+'_X.csv', X, delimiter=",",header=headerX,fmt="%s")

    headerY = ""
    for i in range(16):
        headerY += 'Y_'+str(i)+','
    np.savetxt(suffix+'_Y.csv', Y, delimiter=",",header=headerY,fmt="%s")

    headerL = 'LABEL'
    np.savetxt(suffix+'_L.csv', L, delimiter=",",header=headerL,fmt="%s")
    print "Files saved in", os.getcwd()

def saveHDF(F,X,Y,L,suffix):
    # Create HDF file
    store = pd.HDFStore(suffix+'.h5')
    # Transform F into a pandas series
    sF = pd.Series(F[:,0])
    # Transform X into a pandas dataframe
    headerX = []
    for i in range(256**2):
        headerX.append('X'+str(i))
    dX = pd.DataFrame(X, columns=headerX)
    # Transform Y into a pandas dataframe
    headerY = []
    for i in range(16):
        headerY.append('Y'+str(i))
    dY = pd.DataFrame(Y, columns=headerY)
    # Transform L into a pandas series
    sL = pd.Series(L[:,0])
    # Write pandas variables within the file
    store['F'] = sF
    store['X'] = dX
    store['Y'] = dY
    store['L'] = sL
    # Close HDF
    store.close()
    print "Files saved in "+suffix+'.h5'

def chunkAndBuild(jsonInput, imagesPath, maxN):
    # This function is used to chunk the building of data matrices considering a number of starting images maxN
    # First two inputs are the same of the function "buildInputmatrix". More the one HDF file will be created containing
    # data relative to a number of images equal to maxN*8 (considering all transformations).
    with open(jsonInput) as data_file:
        data = json.load(data_file)
    for c in range(0,len(data),maxN)[:-1]:
        with open(jsonInput[:-5]+'_'+str(c)+'_'+str(c+maxN)+'.json', 'w') as outfile:
            json.dump(data[c:c+maxN], outfile)
        filename = jsonInput[:-5]+'_'+str(c)+'_'+str(c+maxN)+'.json'
        F,X,Y,L = buildInputmatrix(filename, imagesPath)
        suffix = filename[:-5]
        saveHDF(F, X, Y, L, suffix)
    start = range(0,len(data),maxN)[-1]
    with open(jsonInput[:-5]+'_'+str(start)+'_'+str(len(data))+'.json', 'w') as outfile:
        json.dump(data[start:], outfile)
    filename = jsonInput[:-5]+'_'+str(start)+'_'+str(len(data))+'.json'
    F,X,Y,L = buildInputmatrix(filename, imagesPath)
    suffix = filename[:-5]
    saveHDF(F, X, Y, L, suffix)


if __name__ == '__main__':
    if len(sys.argv)==3:
        print 'imagesPath = None'
        imagesPath = None
        maxN = int(sys.argv[2])
    else:
        imagesPath = sys.argv[2]
        maxN = int(sys.argv[3])
    jsonInput = sys.argv[1]
    chunkAndBuild(jsonInput, imagesPath, maxN)
