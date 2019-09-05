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
DATASET_PATH = basePath(os.getenv('DATASET_PATH'))
DATASET_PATH = basePath(os.getenv('DATASET_PATH'))

DELAY = 40 # ms
MIN_AREA = 500 # height * width (px)
MAX_AREA = 2200
RECT_COLOR = (0, 255, 0)

if __name__ == "__main__":
    print(basePath())
    print(BASE_DIR)
    print(VIDEO_PATH)
    print(DATASET_PATH)