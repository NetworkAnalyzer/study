# -*- coding: UTF-8 -*-

import numpy as np
from study.util.array import mean
import study.twmeggs.anfis.anfis as anfis
import study.twmeggs.anfis.membership.membershipfunction as mf
from study import const
from study import video
import csv
from study.util.array import mean
from study.exception import InvalidEpochsValueError


class Anfis:
    def __init__(self, dataset_path, k=5, i=0):
        dataset = self.loadDataset(dataset_path)
        train_data, test_data = self.splitDataset(dataset[:, 1:], k, i)

        last = train_data.shape[1] - 1
        train_x = train_data[:, 0:last]
        train_y = train_data[:, last]
        test_x = test_data[:, 0:last]
        test_y = test_data[:, last]

        self.mean = [
            round(mean(train_x)),
            round(mean(test_x))
        ]

        print('train data: ', train_x.shape[0])
        print('test data:  ', test_x.shape[0])
        print()

        self.mfc = mf.MemFuncs(self.generateMf())
        self.anfis = anfis.ANFIS(train_x, train_y, test_x, test_y, self.mfc)

    def loadDataset(self, path):
        return np.loadtxt(path, delimiter=',')

    def splitDataset(self, dataset, k, i):
        """
        dataset: array
        k: 交差検証の分割数
        i: 交差検証の回数 (何回目か)
        """

        indices = range(dataset.shape[0])
        test_indices = np.array_split(indices, k)[i]

        train_data = []
        test_data = []

        for index in indices:
            if index in test_indices:
                test_data.append(dataset[index])
            else:
                train_data.append(dataset[index])

        return np.array(train_data), np.array(test_data)

    def generateMf(self):
        return [
            # TODO: 複数の特徴量を同時に使用する場合にも対応する
            [
                ['gaussmf', {'mean': self.mean[0], 'sigma': 10}], # video3は160
                ['gaussmf', {'mean': self.mean[1], 'sigma': 7}],  # video3は170
            ],
        ]

    def train(self, epochs):
        epochs = int(epochs)

        if epochs < 2:
            raise InvalidEpochsValueError("you need to set epochs more than 2")

        self.anfis.train(epochs=epochs)

    def plotMF(self, x, inputNumber):
        from skfuzzy import control as ctrl

        self.anfis.plotMF(ctrl.Antecedent(x, 'T').universe, inputNumber)

    def plotResult(self):
        self.anfis.plotResults()

    def plotErrors(self):
        self.anfis.plotErrors()

    def getMinError(self):
        return self.anfis.min_error


def printResult(anfises, name):
    
    accuracy = 0
    precision = 0
    recall = 0

    for anfis in anfises:
        accuracy += anfis.anfis.accuracy
        precision += anfis.anfis.precision
        recall += anfis.anfis.recall

    print('{0}────────────────────────'.format(name))
    print('accuracy:  {0}'.format((accuracy / int(const.get('K')))))
    print('precision: {0}'.format(round(precision / int(const.get('K')), 4)))
    print('recall:    {0}'.format(round(recall / int(const.get('K')), 4)))
    print()


def main(dataset_paths, epochs):

    cars = []
    trucks = []

    for i in range(int(const.get('K'))):
        print('car[{0}]────────────────────────'.format(i))
        cars.append(Anfis(dataset_paths['car']))
        cars[i].train(epochs)
        print('time: {0}s\n'.format(cars[i].anfis.time))

        print('truck[{0}]──────────────────────'.format(i))
        trucks.append(Anfis(dataset_paths['truck']))
        trucks[i].train(epochs)
        print('time: {0}s\n'.format(trucks[i].anfis.time))

        video.main(
            path=const.get('VIDEO_PATH'),
            classify=True,
            anfises={'car' : cars[0].anfis, 'truck' : trucks[0].anfis}
        )

    printResult(cars, 'car')
    printResult(trucks, 'truck')
