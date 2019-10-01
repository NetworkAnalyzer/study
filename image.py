# -*- coding: UTF-8 -*-

import cv2
from object import Object
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

    def __cvt2Gray(self, frame):
        if frame is None:
            return frame

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def subtract(self, before=None, current=None):
        if before is not None:
            self.before_gray = before
        if current is not None:
            self.current_gray = current

        cv2.accumulateWeighted(self.current_gray, self.before_gray, 0.5)
        mdframe = cv2.absdiff(self.current_gray, cv2.convertScaleAbs(self.before_gray))
        return cv2.threshold(mdframe, 3, 255, cv2.THRESH_BINARY)[1]

    def findContours(self, image):
        return cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
if __name__ == "__main__":
    original = cv2.imread(const.IMAGE_PATH)
    filterd  = image.gaborFilter(original, degree=70)

    image.show('gabor filter', filterd, gray=True)