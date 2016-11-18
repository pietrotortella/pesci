import json
from scipy import misc
import os

#---------------------------------------------------------------------------------------------------------------------
# MIRRORING FUNCTION
# This function takes as arguments an "annotation" dictionary and the shapes of the image. It calculates alla the three
# mirrorings on the coordinates.
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

        return newdict_vm, newdict_hvm, newdict_hm

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

        return newdict_vm, newdict_hvm, newdict_hm

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
# MAIN SCRIPT

# Importing coordinates from json file
with open('annotazioni.json') as data_file:
    data = json.load(data_file)

data_copy=[] # this list will contain all the new generated dictionaries
for dict_it in data: # iterator in the main dictionary list

    # Selection and generation of the file names
    fname = os.path.basename(dict_it["filename"])
    ffolder = os.path.dirname(dict_it["filename"])

    image = misc.imread(dict_it["filename"])
    s0, s1 = image.shape[0], image.shape[1]

    fname_mir_vm = os.path.join(ffolder,'mirrored',fname[:-4]+'_mir2.jpg')
    fname_mir_hm = os.path.join(ffolder, 'mirrored', fname[:-4] + '_mir4.jpg')
    fname_mir_hvm = os.path.join(ffolder, 'mirrored', fname[:-4] + '_mir3.jpg')
    fname_col1 = os.path.join(ffolder, '_color', fname[:-4] + '_col1.jpg')
    fname_col2 = os.path.join(ffolder, '_color', fname[:-4] + '_col2.jpg')
    fname_noise = os.path.join(ffolder, '_color', fname[:-4] + '_noise.jpg')

    # Lists containg all the transformed dictionaries of a given image
    dictlist_vm = []
    dictlist_hm = []
    dictlist_hvm = []
    for d in dict_it["annotations"]:
        newdict_vm, newdict_hvm, newdict_hm = mirroring(d,s0,s1)
        dictlist_vm.append(newdict_vm) # lists containing each one the mirrored (one mirror only) dictionaries of a given image
        dictlist_hm.append(newdict_hm)
        dictlist_hvm.append(newdict_hvm)

    dictfile_vm = {
        "annotations": dictlist_vm,
        "class": "image",
        "filename": fname_mir_vm
    }

    dictfile_hm = {
        "annotations": dictlist_hm,
        "class": "image",
        "filename": fname_mir_hm
    }

    dictfile_hvm = {
        "annotations": dictlist_hvm,
        "class": "image",
        "filename": fname_mir_hvm
    }

    dictfile_col1 = {
        "annotations": dict_it["annotations"],
        "class": "image",
        "filename": fname_col1
    }

    dictfile_col2 = {
        "annotations": dict_it["annotations"],
        "class": "image",
        "filename": fname_col2
    }

    dictfile_noise = {
        "annotations": dict_it["annotations"],
        "class": "image",
        "filename": fname_noise
    }

    data_copy.append(dictfile_vm)
    data_copy.append(dictfile_hm)
    data_copy.append(dictfile_hvm)
    data_copy.append(dictfile_col1)
    data_copy.append(dictfile_col2)
    data_copy.append(dictfile_noise)

# Creation of the new json file
with open('annotazioni_out.json', 'w') as f:
    json.dump(data_copy, f, indent=4)
