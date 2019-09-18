# -*- coding: UTF-8 -*-

import numpy as np
import anfis.anfis as anfis
import anfis.membership.membershipfunction as mf
import const

class Anfis:
    def __init__(self, dataset_path, k=5, i=0):
        dataset = self.loadDataset(dataset_path)
        indeces = range(dataset.shape[0])
        test_indeces = np.array_split(indeces, k)[i]
        
        train_data = []
        test_data = []
        for index in indeces:
            if index in test_indeces:
                test_data.append(dataset[index])
            else:
                train_data.append(dataset[index])

        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_x = train_data[:,0:2]
        train_y = train_data[:,2]
        test_x = test_data[:,0:2]
        test_y = test_data[:,2]

        self.mfc = mf.MemFuncs(self.generateMf())
        
        self.anfis = anfis.ANFIS(train_x, train_y, test_x, test_y, self.mfc)

    def loadDataset(self, path):
        return np.loadtxt(path, delimiter=',',usecols=[1,2,3])

    def generateMf(self):
        mf = [
            # ここで大きな配列が3つあるのは，入力が3つだから．入力が増えれば配列も増える
            # 入力の順番と配列の順番は対応していて，入力に近い値がmeanに入っている
            # たとえば，1番目の入力は1以下だから1以下の数値が，3番目の入力は100以下だから100以下の数値が入っている
            [
                # 'gaussmf'はメンバシップ関数名，meanは中心値，sigmaは幅
                # 問題は，なぜ1つの入力に4つのmfが定義されているのか
                ['gaussmf',{'mean':0.5,'sigma':1.}],
                ['gaussmf',{'mean':0.5,'sigma':2.}],
                ['gaussmf',{'mean':0.5,'sigma':10.}],
                ['gaussmf',{'mean':0.5,'sigma':7.}]
            ],
            [
                ['gaussmf',{'mean':2.8,'sigma':2.}],
                ['gaussmf',{'mean':1.5,'sigma':3.}],
                ['gaussmf',{'mean':1.5,'sigma':10.}],
                ['gaussmf',{'mean':1.5,'sigma':5.}]
            ],
        ]

        return mf

    def train(self, epochs=20):
        # epochs >= 2
        self.anfis.train(epochs=int(epochs))

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
    def mean(array):
        return float(sum(array)) / len(array)

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
    
        
