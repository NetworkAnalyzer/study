# -*- coding: UTF-8 -*-

from dotenv import load_dotenv
import os


def _get_base_path(path=None):
    if path is None:
        return BASE_DIR

    return os.path.join(BASE_DIR, path)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(_get_base_path('.env.example'))

VIDEO_PATH = _get_base_path(os.getenv('VIDEO_PATH'))
IMAGE_PATH = _get_base_path(os.getenv('IMAGE_PATH'))
CASCADE_PATH = _get_base_path(os.getenv('CASCADE_PATH'))
K = os.getenv('K')
EPOCHS = os.getenv('EPOCHS')

DELAY = 40  # ms
MIN_AREA = 500  # height * width (px)
MAX_AREA = 9200

# (B, G, R)
RECT_COLOR_CAR = (0, 255, 0)
RECT_COLOR_TRUCK = (0, 0, 255)


def main():
    print(_get_base_path())
    print(BASE_DIR)
    print(VIDEO_PATH)
