# -*- coding: UTF-8 -*-

import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

def show(name, image, gray=False):
    plt.title(name)
    plt.imshow(image)

    if gray:
        plt.gray()
        
    plt.show()

def cvt2Gray(frame):
    if frame is None:
        return frame
    
    if 3 not in frame.shape:
        return frame

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def subtract(before=None, current=None):
    cv2.accumulateWeighted(current, before, 0.5)
    mdframe = cv2.absdiff(current, cv2.convertScaleAbs(before))
    return cv2.threshold(mdframe, 3, 255, cv2.THRESH_BINARY)[1]

def findContours(image):
    return cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def gaborFilter(image, degree=0):
    filter = cv2.getGaborKernel((20, 20), 4.0, np.radians(degree), 10, 0.5, 0)
    gray = cvt2Gray(image)

    return cv2.filter2D(gray, -1, filter)

def glcm(image, degree=0):
    gray = cvt2Gray(image)
    glcm = greycomatrix(gray, [1], [degree])

    features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    values = []
    for feature in features:
        values.append(greycoprops(glcm, feature)[0][0])

    return values