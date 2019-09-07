# -*- coding: UTF-8 -*-

from dotenv import load_dotenv
import os

def basePath(path=None):
    if path is None:
        return BASE_DIR

    return os.path.join(BASE_DIR, path)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(basePath('.env'))

VIDEO_PATH = basePath(os.getenv('VIDEO_PATH'))
DATASET_PATH_FOR_CAR = basePath(os.getenv('DATASET_PATH_FOR_CAR'))
DATASET_PATH_FOR_TRACK = basePath(os.getenv('DATASET_PATH_FOR_TRACK'))
K = os.getenv('K')

DELAY = 40 # ms
MIN_AREA = 500 # height * width (px)
MAX_AREA = 2200
RECT_COLOR = (0, 255, 0)

if __name__ == "__main__":
    print(basePath())
    print(BASE_DIR)
    print(VIDEO_PATH)
    print(DATASET_PATH_FOR_TRACK)