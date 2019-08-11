# -*- coding: UTF-8 -*-

import cv2
import matplotlib.pyplot as plt

class Image:
    def cvt2Gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def show(self, name, image):
        plt.title(name)
        plt.imshow(image, cmap='gray')
        plt.show()
