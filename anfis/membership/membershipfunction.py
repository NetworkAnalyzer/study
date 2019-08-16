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
        # len(rowInout)    ==> 2
        # rowInput ==> [x1, x2]
        # len(self.MFList) ==> 2
        # self.MFList ==> [[mf, mf, mf, mf], [mf, mf, mf, mf]]
        # 同じ 2 でも意味合いが違う
        # MFが8個あるように思うが，実は2個なのか？じゃあなんで1個のMFにつき4つのmeanとsigmaが必要なのか？
        # このコードでは，メンバシップ関数の数と入力の数が合ってないと，メンバシップ関数を適用できないみたい
        if len(rowInput) != len(self.MFList):
            print("Number of variables does not match number of rule sets")

        for i in range(len(rowInput)):
            for k in range(len(self.MFList[i])):
                # self.funcDict[self.MFList[i][k][0]] は関数名gaussmf
                # gaussmf()
                self.funcDict[self.MFList[i][k][0]](rowInput[i],**self.MFList[i][k][1])

        return [[self.funcDict[self.MFList[i][k][0]](rowInput[i],**self.MFList[i][k][1]) for k in range(len(self.MFList[i]))] for i in range(len(rowInput))]