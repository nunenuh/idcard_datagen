import random

import numpy as np

__all__ = ['scale_size_ratio', 'rand_gen', 'random_safe_box_location']


def scale_size_ratio(back_img_size: tuple, top_img_size: tuple,  ratio: float = 0.5, rtype: str = 'w'):
    hbk, wbk = back_img_size
    hbs, wbs = top_img_size
    hbk_ratio, wbk_ratio = ratio * hbk, ratio * wbk
    if rtype == 'w':
        vratio = wbk_ratio
        vratio = vratio / wbs
    elif rtype == 'h':
        vratio = hbk_ratio
        vratio = vratio / wbs

    wnbs, hnbs = vratio * wbs, vratio * hbs
    return int(hnbs), int(wnbs)


def rand_gen(max_val: int, limit_val: int):
    repeat, result = True, max_val
    while repeat:
        result = random.randint(0, max_val)
        if (result - limit_val) > 0: repeat = False
    return result


def random_safe_box_location(back_size: tuple, base_size: tuple):
    hbk, wbk = back_size
    hbs, wbs = base_size

    xmax, ymax = rand_gen(wbk, wbs), rand_gen(hbk, hbs)
    xmin, ymin = xmax - wbs, ymax - hbs

    return xmin, ymin, xmax, ymax
