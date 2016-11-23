from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_images(ims, fish_coords, nofish_coords, fishes, nofishes, heads, tails, ufins, lfins):
    figs = dict()
    for k in range(len(ims)):
        figs[k] = plt.figure()
        ax1 = figs[k].add_subplot(121)
        plt.imshow(ims[k])
        ax1.add_patch(
            patches.Rectangle(
                (fish_coords[k][0], fish_coords[k][1]),
                fish_coords[k][2],
                fish_coords[k][3],
                fill=False,
                color='yellow'
            )
        )
        ax1.add_patch(
            patches.Rectangle(
                (nofish_coords[k][0], nofish_coords[k][1]),
                nofish_coords[k][2],
                nofish_coords[k][3],
                fill=False,
                color='orange'
            )
        )
        plt.scatter(heads[k][0] + fish_coords[k][0],
                    heads[k][1] + fish_coords[k][1],
                    marker='o', edgecolors='r')
        plt.scatter(tails[k][0] + fish_coords[k][0],
                    tails[k][1] + fish_coords[k][1],
                    marker='o', edgecolors='r')
        plt.scatter(ufins[k][0] + fish_coords[k][0],
                    ufins[k][1] + fish_coords[k][1],
                    marker='o', edgecolors='r')
        plt.scatter(lfins[k][0] + fish_coords[k][0],
                    lfins[k][1] + fish_coords[k][1],
                    marker='o', edgecolors='r')

        plt.subplot(122)
        plt.imshow(fishes[k])
        plt.scatter(heads[k][0], heads[k][1], marker='o', edgecolors='r')
        plt.scatter(tails[k][0], tails[k][1], marker='o', edgecolors='r')
        plt.scatter(ufins[k][0], ufins[k][1], marker='o', edgecolors='r')
        plt.scatter(lfins[k][0], lfins[k][1], marker='o', edgecolors='r')

    return figs




def samples_from_dicts(dict_list, return_absolute=False):
    """


    :param dict_list: a list of dictionaries with keys
        'filename', 'annotations'
    :param return_absolute: if true returns also a list with
        the original imaga and the relative positions of the crops
    :return:
    """
    fish_list = []
    nofish_list = []
    head_list = []
    tail_list = []
    ufin_list = []
    lfin_list = []
    fish_coord_list = []
    nofish_coord_list = []
    original_ims = []

    figs = dict()
    for e, d in enumerate(dict_list):
        im = misc.imread(d['filename'])
        annotations = d['annotations']
        fish = None
        nofish = None
        head = None
        tail = None
        ufin = None
        lfin = None

        for note in annotations:
            if note['class'] == 'fish':
                x = note['x']
                y = note['y']
                w = note['width']
                h = note['height']
                fish = im[y: y + h, x:x + w]
                fish_coord = (x, y, w, h)

            elif note['class'] == 'non_fish':
                x = note['x']
                y = note['y']
                w = note['width']
                h = note['height']
                nofish = im[y:y + h, x:x + w]
                nofish_coord = (x, y, w, h)
            elif note['class'] == 'head':
                head = [note['x'], note['y']]
            elif note['class'] == 'tail':
                tail = [note['x'], note['y']]
            elif note['class'] == 'up_fin':
                ufin = [note['x'], note['y']]
            elif note['class'] == 'low_fin':
                lfin = [note['x'], note['y']]

        head[0] = head[0] - fish_coord[0]
        head[1] = head[1] - fish_coord[1]
        tail[0] = tail[0] - fish_coord[0]
        tail[1] = tail[1] - fish_coord[1]
        ufin[0] = ufin[0] - fish_coord[0]
        ufin[1] = ufin[1] - fish_coord[1]
        lfin[0] = lfin[0] - fish_coord[0]
        lfin[1] = lfin[1] - fish_coord[1]

        fish_list.append(fish)
        nofish_list.append(nofish)
        head_list.append(head)
        tail_list.append(tail)
        ufin_list.append(ufin)
        lfin_list.append(lfin)
        fish_coord_list.append(fish_coord)
        nofish_coord_list.append(nofish_coord)
        original_ims.append(im)

#    plt.show()

    if return_absolute:
        returns = (fish_list, nofish_list, head_list, tail_list, ufin_list,
                   lfin_list, fish_coord_list, nofish_coord_list, original_ims)
    else:
        returns = fish_list, nofish_list, head_list, tail_list, ufin_list, lfin_list

    return returns


if __name__ == '__main__':
    import json

    filepath = '/home/terminale11/kaggle_fish/pesci/good_annotations/small-no_errors.json'

    with open(filepath, 'r') as inf:
        data = json.load(inf)

    data = data[:4]
#    print(data[0])
    print(data[0]['annotations'])

    samples = samples_from_dicts(data, return_absolute=True)

    show_images(samples[-1], samples[6], samples[7], samples[0], samples[1], samples[2], samples[3], samples[4], samples[5])








