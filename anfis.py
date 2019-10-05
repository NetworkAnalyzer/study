# -*- coding: UTF-8 -*-

import numpy as np
import anfis.anfis as anfis
import anfis.membership.membershipfunction as mf
import const
import csv
from util.array import mean
from exception import InvalidEpochsValueError

class Anfis:
    def __init__(self, dataset_path, k=5, i=0):
        dataset = self.loadDataset(dataset_path)
        train_data, test_data = self.splitDataset(dataset[:, 1:], k, i)
        
        last = train_data.shape[1] - 1
        train_x = train_data[:,0:last]
        train_y = train_data[:,last]
        test_x  = test_data[:,0:last]
        test_y  = test_data[:,last]

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
        # TODO: 入力する値の数に合わせてMFを生成する

        # for_car
        # mf = [
        #     [
        #         ['gaussmf',{'mean':0.05,'sigma':10.}],
        #         ['gaussmf',{'mean':0.06,'sigma':7.}]
        #     ],[
        #         ['gaussmf',{'mean':1,'sigma':3.}],
        #         ['gaussmf',{'mean':2,'sigma':10.}],
        #     ]
        # ]
        
        # for_c
        mf = [
            [
                ['gaussmf',{'mean':160,'sigma':100.}],
                ['gaussmf',{'mean':170,'sigma':70.}]
            ],
            [
                ['gaussmf',{'mean':10,'sigma':3.}],
                ['gaussmf',{'mean':8,'sigma':10.}],
            ],
            [
                ['gaussmf',{'mean':0.2,'sigma':3.}],
                ['gaussmf',{'mean':0.3,'sigma':10.}],
            ],
            [
                ['gaussmf',{'mean':0.002,'sigma':3.}],
                ['gaussmf',{'mean':0.001,'sigma':10.}],
            ],
            [
                ['gaussmf',{'mean':1,'sigma':3.}],
                ['gaussmf',{'mean':0.9,'sigma':10.}],
            ],
        ]

        return mf

    def train(self, epochs=20):
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

if __name__ == "__main__":
    def putResult(anfis):
        errors = [[round(error, 4)] for i, error in enumerate(anfis.anfis.errors)]

        with open('result_{0}.csv'.format(const.EPOCHS), 'w') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerows(errors)

    cars = []
    trucks = []

    cars_accuracy = []
    cars_precision = []
    cars_recall = []
    trucks_accuracy = []
    trucks_precision = []
    trucks_recall = []

    for i in range(int(const.K)):
        cars.append(Anfis(const.DATASET_PATH_FOR_CAR))
        trucks.append(Anfis(const.DATASET_PATH_FOR_TRUCK))

        cars[i].train(epochs=const.EPOCHS)
        print('car_{0}: {1}s'.format(i, cars[i].anfis.time))
        trucks[i].train(epochs=const.EPOCHS)
        print('truck_{0}: {1}s'.format(i, trucks[i].anfis.time))

        cars_accuracy.append(cars[i].anfis.accuracy)
        cars_precision.append(cars[i].anfis.precision)
        cars_recall.append(cars[i].anfis.recall)
        
        trucks_accuracy.append(trucks[i].anfis.accuracy)
        trucks_precision.append(trucks[i].anfis.precision)
        trucks_recall.append(trucks[i].anfis.recall)
    
    print('car────────────────────────')
    print('accuracy:{0}'.format(mean(cars_accuracy)))
    print('precision:{0}'.format(mean(cars_precision)))
    print('recall:{0}'.format(mean(cars_recall)))

    print('truck──────────────────────')
    print('accuracy:{0}'.format(mean(trucks_accuracy)))
    print('precision:{0}'.format(mean(trucks_precision)))
    print('recall:{0}'.format(mean(trucks_recall)))
