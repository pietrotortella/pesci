from scipy import misc
import matplotlib.pyplot as plt

import json

from samples_from_json import samples_from_dicts, show_images
from create_passport_photos import get_passport_pic

filepath = '/home/terminale11/kaggle_fish/pesci/good_annotations/small-no_errors.json'

with open(filepath, 'r') as inf:
    data = json.load(inf)

data = data[:5]

(fishes, nofishes, heads, tails, ufins, lfins,
 fish_coords, nofish_coords, bigimages) = samples_from_dicts(data, return_absolute=True)


figs_orig = show_images(bigimages, fish_coords, nofish_coords,
                        fishes, nofishes, heads, tails, ufins, lfins)

passports = dict()

figs = dict()
for k in range(len(data)):
    passports[k] = get_passport_pic(fishes[k], heads[k], tails[k], ufins[k], lfins[k])
    figs[k] = plt.figure()
    plt.imshow(passports[k], cmap=plt.cm.gray)

plt.show()