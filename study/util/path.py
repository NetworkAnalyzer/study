# -*- coding: UTF-8 -*-

import os


def get_base_path(path=None):

    CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
    BASE_DIR = os.path.join(CURRENT_DIR, '..')

    if path is None:
        return BASE_DIR

    return os.path.join(BASE_DIR, path)
