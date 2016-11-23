import numpy as np

class FishClass():

    fishType = ""
    fishPixels = np.empty(65536, dtype=int)
    imageName = ""
    fish_X = 0
    fish_Y = 0
    fish_H = 0
    fish_W = 0

    nonfish_X = 0
    nonfish_Y = 0
    nonfish_H = 0
    nonfish_W = 0

    head_X = 0
    head_Y = 0

    tail_X = 0
    tail_Y = 0

    upfin_X = 0
    upfin_Y = 0

    lowfin_X = 0
    lowfin_Y = 0
    original_width = 0
    original_heigth = 0

    def __init__(self): # constructor - optional in Python
        self.imageName = ""


