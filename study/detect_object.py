# -*- coding: UTF-8 -*-

import glob
import csv
from study.object import Object
from study.image import Image


def main(video_path):
    paths = glob.glob("image/*.png")
    for dataset_for in ['c', 't']:
        cnt = 1
        data = []
        for path in paths:
            image = Image(path)
            object = Object(0, 0, image.height, image.width, image.image)

            data.append(
                [
                    cnt,
                    round(object.contrast, 4),
                    round(object.dissimilarity, 4),
                    round(object.homogeneity, 4),
                    round(object.asm, 4),
                    round(object.correlation, 4),
                    _getAns(path, dataset_for),
                ]
            )

            cnt += 1
        
        file_name = 'dataset/dataset_for_{0}.csv'.format(dataset_for)
        
        with open(file_name, 'w') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerows(data)

        print(file_name + 'is generated')