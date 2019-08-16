# -*- coding: UTF-8 -*-

import numpy as np
import anfis.anfis as anfis
import anfis.membership.membershipfunction as mf
import const

if __name__ == "__main__":
    def loadDataset(path):
        dataset = np.loadtxt(path, usecols=[1,2,3,4])
        train_data = dataset[:,0:3]
        test_data = dataset[:,3]

        return train_data, test_data

    def defineMF():
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

    train_data, test_data = loadDataset(const.DATASET_PATH)
    mfc = mf.MemFuncs(defineMF())
    anfis1 = anfis.ANFIS(train_data, test_data, mfc)

    anfis1.trainHybridJangOffLine(epochs=20)
    anfis1.plotResults()