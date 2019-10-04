# -*- coding: UTF-8 -*-

import util.image as image
from skimage.feature import greycomatrix, greycoprops

class Object:
    def __init__(self, x, y, w, h, image):
        # information
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.image = image

        # features
        self.glcm = self.__glcm()
        self.contrast      = greycoprops(self.glcm, 'contrast')[0][0]
        self.dissimilarity = greycoprops(self.glcm, 'dissimilarity')[0][0]
        self.homogeneity   = greycoprops(self.glcm, 'homogeneity')[0][0]
        self.asm           = greycoprops(self.glcm, 'ASM')[0][0]
        self.correlation   = greycoprops(self.glcm, 'correlation')[0][0]

    def compactness(self):
        area = self.h * self.w
        prim = 2 * self.h + 2 * self.w

        return float(area) / prim**2

    def hwr(self):
        return float(self.h) / self.w

    def __glcm(self):
        grey = image.cvt2Gray(self.image)
        return greycomatrix(grey, [1], [0])
