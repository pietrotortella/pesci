import math

import numpy as np

from skimage.color import rgb2grey
from skimage.transform import rotate, resize


def x_mirror(im, head, tail, ufin, lfin):
    """
    creates the mirror data mirrored along the x-axis, around
    the middle row of the image
    :param im: numpy.array, the image
    :param head: a pair of ints, a point in im
    :param tail: a pair of ints, a point in im
    :param ufin: a pair of ints, a point in im
    :param lfin: a pair of ints, a point in im
    :return: the same data as the parameters
    """

    imret = np.array(im[::-1, :])
    headret = (im.shape[0] - head[0], head[1])
    tailret = (im.shape[0] - tail[0], tail[1])
    ufinret = (im.shape[0] - ufin[0], ufin[1])
    lfinret = (im.shape[0] - lfin[0], lfin[1])

    return imret, headret, tailret, ufinret, lfinret


def y_mirror(im, head, tail, ufin, lfin):
    """
    creates the mirror data mirrored along the y-axis, around
    the middle column of the image
    :param im: numpy.array, the image
    :param head: a pair of ints, a point in im
    :param tail: a pair of ints, a point in im
    :param ufin: a pair of ints, a point in im
    :param lfin: a pair of ints, a point in im
    :return: the same data as the parameters
    """

    imret = np.array(im[:, :-1])
    headret = (head[0], im.shape[1] - head[1])
    tailret = (tail[0], im.shape[1] - tail[1])
    ufinret = (ufin[0], im.shape[1] - ufin[1])
    lfinret = (lfin[0], im.shape[1] - lfin[1])

    return imret, headret, tailret, ufinret, lfinret


def get_passport_pic(im, head, tail, ufin, lfin, newshape=(64, 64)):
    """
    Transform the image to obtain a passport-like picture.
    This consist in transforming the image to greyscale,
    mirroring it to obtain the head of
    the fish to the right and the upper fin in the upper part of the
    picture, rotating the image to have the head-tail axes horizontal
    and resizing the image to the appropriate shape

    :param im: numpy.ndarray, the image
    :param head: a pair of ints, the coordinates of the head
    :param tail: a pair of ints, the coordinates of the tail
    :param ufin: a pair of ints, the coordinates of the upper fin
    :param lfin: a pair of ints, the coordinates of the lower fin
    :param newshape: pair of ints, the shape of the image to return
    :return: a numpy.ndarray, the transformed image
    """

    proc_im = np.array(im)
    proc_im = rgb2grey(proc_im)

    head_is_right = head[1] > tail[1]
    ufin_is_up = ufin[0] < lfin[0]
    if not head_is_right:
        proc_im, head, tail, ufin, lfin = y_mirror(proc_im, head, tail, ufin, lfin)

    if not ufin_is_up:
        proc_im, head, tail, ufin, lfin = x_mirror(proc_im, head, tail, ufin, lfin)

    slope = (head[0] - tail[0]) / (head[1] - tail[1])
    theta = math.atan(slope)
    theta = math.degrees(theta)

    proc_im  = rotate(proc_im, angle=theta)
    proc_im = resize(proc_im, newshape)

    return proc_im





