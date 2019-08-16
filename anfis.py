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
        dataset = np.loadtxt(path, usecols=[1,2,3,4])
        train_data = dataset[:,0:3]
        test_data = dataset[:,3]

        return train_data, test_data

    def generateMf(self):
        mf = [
            [
                # 'gaussmf'はメンバシップ関数名，meanは中心値，sigmaは幅
                ['gaussmf',{'mean':0.4,'sigma':1.}],
                ['gaussmf',{'mean':1.,'sigma':2.}],
                ['gaussmf',{'mean':0.8,'sigma':10.}],
                ['gaussmf',{'mean':0.6,'sigma':7.}]
            ],
            [
                ['gaussmf',{'mean':1.,'sigma':2.}],
                ['gaussmf',{'mean':0.9,'sigma':3.}],
                ['gaussmf',{'mean':1.,'sigma':10.}],
                ['gaussmf',{'mean':1.5,'sigma':5.}]
            ],
            [
                ['gaussmf',{'mean':50.,'sigma':2.}],
                ['gaussmf',{'mean':60.,'sigma':3.}],
                ['gaussmf',{'mean':70.,'sigma':10.}],
                ['gaussmf',{'mean':80,'sigma':5.}]
            ]
        ]

        return mf

    def train(self, epochs=20):
        """
        epochs >= 2
        """
        self.anfis.trainHybridJangOffLine(epochs=epochs)

    def plotResult(self):
        self.anfis.plotResults()

if __name__ == "__main__":
    anfis1 = Anfis(const.DATASET_PATH)
    anfis2 = Anfis(const.DATASET_PATH)
    anfis3 = Anfis(const.DATASET_PATH)

    anfis1.train(epochs=2)
    anfis2.train(epochs=5)
    anfis3.train(epochs=10)

    anfis1.plotResult()
    anfis2.plotResult()
    anfis3.plotResult()