# -*- coding: UTF-8 -*-

import glob
import csv
import os
from study.util.path import dataset_path
from study.object import Object
from study.image import Image


def _getAns(path, dataset_for):
    return 1 if path.find('_{0}.'.format(dataset_for)) is not -1 else 0


def main(video_name, feature_name):

    paths = glob.glob("image/{0}/*.png".format(video_name))

    os.makedirs(dataset_path(video_name), exist_ok=True)

    for dataset_for in ['c', 't']:
        cnt = 1
        data = []
        for path in paths:
            image = Image(path)
            object = Object(0, 0, image.height, image.width, image.image)
            features = {
                'contrast'      : round(object.contrast, 4),
                'dissimilarity' : round(object.dissimilarity, 4),
                'homogeneity'   : round(object.homogeneity, 4),
                'asm'           : round(object.asm, 4),
                'correlation'   : round(object.correlation, 4)
            }

            data.append([
                    cnt,
                    features[feature_name],
                    _getAns(path, dataset_for),
            ])

            cnt += 1
        
        file_name = 'dataset_{0}_for_{1}.csv'.format(feature_name, dataset_for)
        print('first 3 lines of dataset ───────────────────────────')
        print(data[0])
        print(data[1])
        print(data[2])
        output_path = dataset_path(video_name + '/' + file_name)
        
        with open(output_path, 'w') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerows(data)

        print(output_path + ' is generated')
        print()
