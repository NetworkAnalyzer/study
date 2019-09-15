# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:41:58 2014

@author: tim.meggs
"""

# skfuzzy = scikit fuzzy
from skfuzzy import gaussmf, gbellmf, sigmf

class MemFuncs:
    'Common base class for all employees'
    funcDict = {'gaussmf': gaussmf, 'gbellmf': gbellmf, 'sigmf': sigmf}


    def __init__(self, MFList):
        self.MFList = MFList

    # rowInput: [mean, sigma]
    def evaluateMF(self, rowInput):
        # rowInput ==> [x1, x2]
        # len(rowInput)    ==> 2
        # self.MFList ==> [[mf, mf, mf, mf], [mf, mf, mf, mf]]
        # len(self.MFList) ==> 2
        if len(rowInput) != len(self.MFList):
            print("Number of variables does not match number of rule sets")

        # x1をMFList[0]のMFすべてに，x2をMFList[1]のMFすべてにそれぞれ適用した結果を得る
        # 1つの入力に対するMFの数は，4つである必要がない
        return [[self.funcDict[self.MFList[i][k][0]](rowInput[i],**self.MFList[i][k][1]) for k in range(len(self.MFList[i]))] for i in range(len(rowInput))]