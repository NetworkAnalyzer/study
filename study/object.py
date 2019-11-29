# -*- coding: UTF-8 -*-

import study.util.image as util_image
from skimage.feature import greycomatrix, greycoprops


class Object:
    def __init__(self, image, h=0, w=0):
        self.compactness = self.__compactness(h, w)
        self.hwr = self.__HWR(h, w)
        self.glcm = self.__glcm(image)

    def __glcm(self, image):
        grey = util_image.cvt2Gray(image)
        return greycomatrix(grey, [1], [0])

    def __compactness(self, h, w):
        area = h * w
        prim = 2 * h + 2 * w

        return float(area) / prim**2
    
    def __HWR(self, h, w):
        return float(h) / w

    def get(self, features):
        ret = []

        for feature in features:
            if feature == 'compactness':
                ret.append(self.compactness)
            elif feature == 'hwr' or feature == 'HWR':
                ret.append(self.hwr)
            else:
                ret.append(self.greycoprops_wrap(self.glcm, feature))
        
        return ret

    def greycoprops_wrap(self, glcm, feature):
        if feature == 'asm':
            feature = 'ASM'
        
        return greycoprops(glcm, feature)[0][0]
