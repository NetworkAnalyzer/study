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
CASCADE_PATH = basePath(os.getenv('CASCADE_PATH'))
DATASET_PATH_FOR_CAR = basePath(os.getenv('DATASET_PATH_FOR_CAR'))
DATASET_PATH_FOR_TRUCK = basePath(os.getenv('DATASET_PATH_FOR_TRUCK'))
K = os.getenv('K')
EPOCHS = os.getenv('EPOCHS')

DELAY = 40 # ms
MIN_AREA = 500 # height * width (px)
MAX_AREA = 9200

# (B, G, R)
RECT_COLOR_CAR = (0, 255, 0)
RECT_COLOR_TRUCK = (0, 0, 255)

if __name__ == "__main__":
    print(basePath())
    print(BASE_DIR)
    print(VIDEO_PATH)
    print(DATASET_PATH_FOR_TRUCK)