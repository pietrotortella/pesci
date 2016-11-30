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

    imret = np.array(im[:, ::-1])
    headret = (head[0], im.shape[1] - head[1])
    tailret = (tail[0], im.shape[1] - tail[1])
    ufinret = (ufin[0], im.shape[1] - ufin[1])
    lfinret = (lfin[0], im.shape[1] - lfin[1])

    return imret, headret, tailret, ufinret, lfinret


def get_passport_pic(im, head, tail, ufin, lfin, newshape=(96, 96)):
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
    #proc_im = rgb2grey(proc_im)

    slope = (head[1] - tail[1]) / (head[0] - tail[0] + 10 ** -8)
    theta = math.atan(slope)
    theta = math.degrees(-theta)


    proc_im  = rotate(proc_im, angle=theta)
    proc_im = resize(proc_im, newshape)



    ver = np.array([head, tail, ufin, lfin])

    x_ver = np.zeros(4)
    y_ver = np.zeros(4)

    theta_rad = - math.atan(slope)


    for i in range(0, 4):
        x_ver[i] = int(round(math.cos(theta_rad) * (ver[i, 0]) - math.sin(theta_rad) * (ver[i, 1])))
        y_ver[i] = int(round(math.sin(theta_rad) * (ver[i, 0]) + math.cos(theta_rad) * (ver[i, 1])))

    x_min = min(x_ver)
    y_min = min(y_ver)

    if x_min < 0:
        x_trasl = -x_min
    else:
        x_trasl = 0

    if y_min < 0:
        y_trasl = -y_min
    else:
        y_trasl = 0

    x_ver = x_ver + x_trasl
    y_ver = y_ver + y_trasl

    head = [x_ver[0],y_ver[0]]
    tail = [x_ver[1],y_ver[1]]
    ufin = [x_ver[2],y_ver[2]]
    lfin = [x_ver[3],y_ver[3]]


    head_is_up = head[0] < tail[0]
    if not head_is_up:
        proc_im, head, tail, ufin, lfin = x_mirror(proc_im, head, tail, ufin, lfin)
    ufin_is_left = ufin[1] < lfin[1]
    if not ufin_is_left:
        proc_im, head, tail, ufin, lfin = y_mirror(proc_im, head, tail, ufin, lfin)


    return proc_im





