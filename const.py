# -*- coding: UTF-8 -*-

from dotenv import load_dotenv
import os

class Const:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    load_dotenv(os.path.join(BASE_DIR, '.env'))

    VIDEO_PATH = os.path.join(BASE_DIR, os.getenv('VIDEO_PATH'))

    DELAY = 40 # ms
    MIN_AREA = 500 # height * width (px)
    RECT_COLOR = (0, 255, 0)

if __name__ == "__main__":
    const = Const()
    print(const.BASE_DIR)
    print(const.VIDEO_PATH)