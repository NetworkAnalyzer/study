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

def cornerHarris(img, blockSize=2, ksize=3, k=0.04):
    """コーナーを検出する

    Parameters
    ----------
    img
        入力画像
    blockSize
        コーナー検出の際に考慮する隣接領域のサイズ
    ksize
        Sobelの勾配オペレータのカーネルサイズ
    k
        式中のフリーパラメータ

    Returns
    -------
    img
        入力画像に検出したコーナーを描画したもの
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(gray, blockSize, ksize, k)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    return img

def pltShow(title, img):
    plt.imshow(img)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    file_name = 'car2'
    input_name = 'result/{0}.png'.format(file_name)
    output_name = 'result/{0}_corner.png'.format(file_name)

    input_path = os.path.join(BASE_DIR, input_name)
    output_path = os.path.join(BASE_DIR, output_name)

    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pltShow(input_name, img)

    img = cornerHarris(img)
    pltShow(output_name, img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()