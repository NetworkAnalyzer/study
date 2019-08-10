# -*- coding: UTF-8 -*-
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

def closing(img):
    ## create kernel for closing
    KERNEL_CLOSING = int(os.getenv('KERENL_CLOSING'))
    kernel = np.ones((KERNEL_CLOSING,KERNEL_CLOSING),np.uint8)

    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def filling_holes(img):

    # mask is larger than img by 2 pixel
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # adopt mask to img
    flags = 4 | 255 << 8 | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(img, mask, seedPoint = (2, 2), newVal = (0, 0, 255), flags=flags)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) # add alpha channel
    mask = mask[1:-1, 1:-1]                      # rm 1px around
    rgba[mask==255] = 255                        # rewrite rgba as 255 if mask is 255
    
    return rgba

def sharpening(img, neighborhood):

    if neighborhood == 4:
        # 4近傍
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], np.float32)
    else:
        # 8近傍
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], np.float32)

    return cv2.filter2D(img, -1, kernel)

if __name__ == '__main__':

    filepath = sys.argv[1]
    img = cv2.imread(filepath)  

    pre1 = closing(img)
    cv2.imwrite('preprocessing1.jpg', pre1)
    print('closing is done')

    # pre2 = filling_holes(img)
    # cv2.imwrite('preprocessing2.jpg', pre2)
    # print('filling hole is done')

    pre3 = sharpening(pre1, os.getenv('NEIGHBORHOOD'))
    cv2.imwrite('preprocessing3.jpg', pre3)
    print('sharpening is done')
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()