# -*- coding: UTF-8 -*-

import numpy as np
import study.twmeggs.anfis.anfis as anfis
import study.twmeggs.anfis.membership.membershipfunction as mf
from study import const
from study import video
from study.util.array import mean, concat
from study.exception import InvalidEpochsValueError


class Anfis:
    def __init__(self, dataset_paths, k=5, i=0):
        
        train_x = []
        test_x = []

        for path in dataset_paths:
            dataset = self.loadDataset(path)
            train_data, test_data = self.splitDataset(dataset[:, 1:], k, i)

            last = train_data.shape[1] - 1
            train_x = concat(train_x, train_data[:, 0:last], 1)
            train_y = train_data[:, last]
            test_x = concat(test_x, test_data[:, 0:last], 1)
            test_y = test_data[:, last]
        
        if const.get('VERBOSE'):
            print('train data: ', train_x.shape[0])
            print('test data:  ', test_x.shape[0])
            print()

        self.mfc = mf.MemFuncs(self.generateMf(train_x, test_x))
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

    def generateMf(self, train_x, test_x):
        mf = []

        for i in range(len(train_x[0])):
            mf1 = mean(train_x[:, i])
            mf2 = mean(test_x[:, i])

            if int(const.get('VIDEO_NUM')) == 1 and const.get('FEATURE')[i] == 'contrast':
                mf1 += 30
                mf2 -= 30

            mf.append([
                ['gaussmf', {'mean' : mf1, 'sigma' : 10}],
                ['gaussmf', {'mean' : mf2, 'sigma' : 7}],
            ])

        return mf

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

def printTime(time):
    print('time: {0}s\n'.format(time))


def calcF_measure(precision, recall):
    return round((2 * precision * recall) / (precision + recall), 4)
    
def printResult(anfis, name):
    
    accuracy  = round(anfis.anfis.accuracy, 4)
    precision = round(anfis.anfis.precision, 4)
    recall    = round(anfis.anfis.recall, 4)
    f_measure = calcF_measure(precision, recall)

    if const.get('VERBOSE'):
        print('{0}───────────────────────────────────'.format(name))
        print('accuracy','precision','recall', '\tf_measure', sep="\t")
        print( accuracy,  precision,  recall, f_measure, sep="\t\t")
        print()
    else:
        print(','.join(const.get('FEATURE')))
        print(const.get('VIDEO_NAME')[int(const.get('VIDEO_NUM'))])
        print(name)
        print(accuracy, f_measure, sep=',')


def main(dataset_paths):

    cars = []
    trucks = []

    for i in range(int(const.get('K'))):
        if const.get('VERBOSE'):
            print('car[{0}]────────────────────────'.format(i))
        cars.append(Anfis(dataset_paths['car'], i=i))
        cars[i].train(const.get('EPOCHS'))
        if const.get('VERBOSE'):
            printTime(cars[i].anfis.time)

        if const.get('VERBOSE'):
            print('truck[{0}]──────────────────────'.format(i))
        trucks.append(Anfis(dataset_paths['truck'], i=i))
        trucks[i].train(const.get('EPOCHS'))
        if const.get('VERBOSE'):
            printTime(trucks[i].anfis.time)


    printResult(cars, 'car')
    printResult(trucks, 'truck')
