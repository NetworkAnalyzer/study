# -*- coding: UTF-8 -*-

import cv2
import matplotlib.pyplot as plt

def show(name, image, gray=False):
    plt.title(name)
    plt.imshow(image)

    if gray:
        plt.gray()
        
    plt.show()

def cvt2Gray(frame):
    if frame is None:
        return frame

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def subtract(before=None, current=None):
    cv2.accumulateWeighted(current, before, 0.5)
    mdframe = cv2.absdiff(current, cv2.convertScaleAbs(before))
    return cv2.threshold(mdframe, 3, 255, cv2.THRESH_BINARY)[1]

def findContours(image):
    return cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
