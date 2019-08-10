# -*- coding: UTF-8 -*-
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from math import sqrt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

def compactness(h, w):
    area = h*w
    prim = 2*h + 2*w
    
    return float(area) / prim**2

def hwr(h, w):
    return float(h) / w

def velocity(c_now, c_next):
    second_per_frame = 1 / float(os.getenv('FPS'))
    c_diff = c_now - c_next
    distance = sqrt(c_diff[0]**2 + c_diff[1]**2)

    return distance / second_per_frame

def extract_features(obj_now, obj_next):
    # ひとまずvelocityは無視
    # return compactness(obj_next[3], obj_next[2]), hwr(obj_next[3], obj_next[2]), velocity(obj_now[4:], obj_next[4:])
    return compactness(obj_next[3], obj_next[2]), hwr(obj_next[3], obj_next[2])

if __name__ == '__main__':
    img = cv2.imread(os.path.join(BASE_DIR, 'gingham-checks.jpeg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    plt.imshow(img)
    plt.title('original image')
    plt.show()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    
    plt.imshow(img)
    plt.title('cornerHarris image')
    plt.show()

    h, w = img.shape[:2]
    c_now  = np.array([40, 40])
    c_next = np.array([40, 42])

    feature1 = compactness(h, w)
    feature2 = hwr(h, w)
    feature3 = velocity(c_now, c_next)

    print(feature1)
    print(feature2)
    print(feature3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()