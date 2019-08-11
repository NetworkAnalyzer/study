# -*- coding: UTF-8 -*-

import cv2
import matplotlib.pyplot as plt

class Image:
    def show(self, name, image):
        plt.title(name)
        plt.imshow(image, cmap='gray')
        plt.show()
