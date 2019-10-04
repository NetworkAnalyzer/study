# -*- coding: UTF-8 -*-

import cv2
import util.image as image
import const
import matplotlib.pyplot as plt


class Image:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def show(self, name, image, gray=False):
        plt.title(name)
        plt.imshow(image)

        if gray:
            plt.gray()

        plt.show()


if __name__ == "__main__":
    original = cv2.imread(const.IMAGE_PATH)
    features = image.glcm(original, degree=0)
    print(features)

    # image.show('gabor filter', filterd, gray=True)
