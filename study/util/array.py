# -*- coding: UTF-8 -*-

import numpy as np

def mean(array):
    return round(float(sum(array)) / len(array), 4)

def concat(src, dst, axis=0):
    if (src == []):
        return dst
    else:
        return np.concatenate([src, dst], axis)
