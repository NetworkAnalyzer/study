# -*- coding: UTF-8 -*-

import os
from study.util.path import base_path, video_path, image_path, cascade_path, dataset_path


const_list = {
    'VIDEO_PATH'         : video_path(),
    'IMAGE_PATH'         : image_path(),
    'CASCADE_PATH'       : cascade_path('car.xml'),
    'DATASET_PATH_CAR'   : dataset_path(),
    'DATASET_PATH_TRUCK' : dataset_path(),
    'K'                  : 1,
    'EPOCHS'             : 20,
    # NOTE: (B, G, R)
    'RECT_COLOR_CAR'     : (0, 255, 0),
    'RECT_COLOR_TRUCK'   : (0, 0, 255),
    'EXT_VIDEO'          : 'mp4',
    'EXT_IMAGE'          : 'png', 
}

def get(key):
    return const_list[key]

def set(key, value):
    const_list[key] = value
    