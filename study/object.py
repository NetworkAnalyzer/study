# -*- coding: UTF-8 -*-

import study.util.image as image
from skimage.feature import greycomatrix, greycoprops


class Object:
    def __init__(self, image):
        self.image = image
        self.glcm = self.__glcm()

    def __glcm(self):

        grey = image.cvt2Gray(self.image)
        return greycomatrix(grey, [1], [0])

    def get(self, feature_name):
        
        return greycoprops(self.glcm, feature_name)[0][0]
