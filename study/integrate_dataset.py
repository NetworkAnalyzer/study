# -*- coding: UTF-8 -*-

import glob
import csv
import os
import numpy as np
from study.util.path import dataset_path
from study.object import Object
from study.image import Image


def main():

    for ans in ['c', 't']:
        for feature in ['contrast', 'dissimilarity', 'homogeneity', 'asm', 'correlation']:
            file_name = "dataset_{feature}_for_{ans}.csv".format(feature=feature, ans=ans)
            paths = glob.glob("study/dataset/**/" + file_name, recursive=True)

            if paths == []:
                print('datasets not found')
                exit()

            cnt = 1
            integrated = []
            for path in paths:
                if path == "study/dataset/" + file_name:
                    continue

                for data in np.loadtxt(path, delimiter=','):
                    integrated.append([
                        cnt,
                        round(data[1], 4),
                        int(data[2]),
                    ])
                    cnt += 1
        
            print('first 3 lines of dataset ───────────────────────────')
            print(integrated[0])
            print(integrated[1])
            print(integrated[2])

            output_path = dataset_path(file_name)
            
            with open(output_path, 'w') as f:
                w = csv.writer(f, lineterminator='\n')
                w.writerows(integrated)

            print(output_path + ' is generated')
            print()
