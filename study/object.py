# -*- coding: UTF-8 -*-

import study.util.image as util_image
from skimage.feature import greycomatrix, greycoprops


class Object:
    def __init__(self, image):
        self.glcm = self.__glcm(image)

    def __glcm(self, image):
        grey = util_image.cvt2Gray(image)
        return greycomatrix(grey, [1], [0])

    def get(self, feature_name):
        return greycoprops(self.glcm, feature_name)[0][0]
