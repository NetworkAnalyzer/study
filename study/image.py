# -*- coding: UTF-8 -*-

import cv2
import study.util.image as image
from study import const
import matplotlib.pyplot as plt


class Image:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]


def main():
    original = cv2.imread(const.IMAGE_PATH)
    features = image.glcm(original, degree=0)
    print(features)
