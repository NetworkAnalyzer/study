# -*- coding: UTF-8 -*-

import numpy as np
import anfis.anfis as anfis
import anfis.membership.membershipfunction as mf
import const

class Anfis:
    def __init__(self, path):
        self.train_data, self.test_data = self.loadDataset(path)
        self.mfc = mf.MemFuncs(self.generateMf())
        self.anfis = anfis.ANFIS(self.train_data, self.test_data, self.mfc)

    def loadDataset(self, path):
        dataset = np.loadtxt(path, delimiter=',',usecols=[1,2,3])
        train_data = dataset[:,0:2]
        test_data = dataset[:,2]

        return train_data, test_data

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
        """
        epochs >= 2
        """
        self.anfis.trainHybridJangOffLine(epochs=epochs)

    def plotMF(self, x, inputNumber):
        from skfuzzy import control as ctrl
        self.anfis.plotMF(ctrl.Antecedent(x, 'T').universe, inputNumber)

    def plotResult(self):
        self.anfis.plotResults()

    def plotErrors(self):
        self.anfis.plotErrors()
    
    def getAccuracyRate(self):
        return self.anfis.getAccuracyRate()

if __name__ == "__main__":
    anfis1 = Anfis(const.DATASET_PATH)
    anfis2 = Anfis(const.DATASET_PATH)
    anfis3 = Anfis(const.DATASET_PATH)

    anfis1.plotMF(range(-30, 30), 1)

    anfis1.train(epochs=2)
    anfis2.train(epochs=5)
    anfis3.train(epochs=10)

    anfis1.plotResult()
    anfis2.plotResult()
    anfis3.plotResult()