from scipy import misc
import matplotlib.pyplot as plt

import json
import numpy as np
import time
# ima = misc.imread('/home/terminale2/Downloads/train/ALB/img_00003.jpg')
# plt.imshow(ima)
# plt.show()
import os


from samples_from_json import samples_from_dicts, show_images
from create_passport_photos import get_passport_pic

filepath = '/home/terminale2/Documents/ALL_small.json'
def samples(filepath):
    with open(filepath, 'r') as inf:
        data = json.load(inf)

    data = data

    (fishes, nofishes, heads, tails, ufins, lfins, labels,
     fish_coords, nofish_coords, bigimages) = samples_from_dicts(data, return_absolute=True)


    # figs_orig = show_images(bigimages, fish_coords, nofish_coords,
    #                         fishes, nofishes, heads, tails, ufins, lfins)

    passports = dict()
    start_time = time.time()
    #figs = dict()



    passports[0] = get_passport_pic(fishes[0], heads[0], tails[0], ufins[0], lfins[0])
    database = passports[0].flatten()
    database = database.reshape((1, len(database)))

    for k in range(1,len(data)):
        passports[k] = get_passport_pic(fishes[k], heads[k], tails[k], ufins[k], lfins[k])
        data = np.ndarray.flatten(passports[k])
        data = data.reshape((1, len(data)))
        database = np.concatenate((database, data), axis=0)

     #   figs[k] = plt.figure()
     #   plt.imshow(passports[k], cmap=plt.cm.gray)
    #
    # plt.show()

    # database = passports[0].flatten()
    # database = database.reshape((1, len(database)))
    # print database.shape
    # for z in range(1, len(data)):
    #     data = np.ndarray.flatten(passports[z])
    #     data = data.reshape((1, len(data)))
    #
    #     database = np.concatenate((database, data), axis=0)


    database=np.array(database)

    print time.time() - start_time


    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( 10, 700))

    return database, labels
#print database