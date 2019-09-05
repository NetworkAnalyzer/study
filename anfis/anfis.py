# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 07:30:34 2014

@author: tim.meggs
"""
import itertools
import numpy as np
from .membership import mfDerivs
import copy
import const

class ANFIS:
    """Class to implement an Adaptive Network Fuzzy Inference System: ANFIS"

    Attributes:
        X
        Y
        XLen
        memClass
        memFuncs
        memFuncsByVariable
        rules
        consequents
        errors
        memFuncsHomo
        trainingType


    """

    def __init__(self, X, Y, mf):
        self.X = np.array(copy.copy(X))
        self.Y = np.array(copy.copy(Y))
        self.XLen = len(self.X)
        self.memClass = copy.deepcopy(mf)
        self.memFuncs = self.memClass.MFList
        self.memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]
        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))
        self.consequents = np.empty(self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1))
        self.consequents.fill(0)
        self.errors = np.empty(0)
        self.min_error = 1
        self.memFuncsHomo = all(len(i)==len(self.memFuncsByVariable[0]) for i in self.memFuncsByVariable)
        self.trainingType = 'Not trained yet'

    # Least Square Estimation 最小2乗推定値
    def LSE(self, A, B, initialGamma = 1000.):
        coeffMat = A
        rhsMat = B
        S = np.eye(coeffMat.shape[1])*initialGamma
        x = np.zeros((coeffMat.shape[1],1)) # need to correct for multi-dim B
        for i in range(len(coeffMat[:,0])):
            a = coeffMat[i,:]
            b = np.array(rhsMat[i])
            S = S - (np.array(np.dot(np.dot(np.dot(S,np.matrix(a).transpose()),np.matrix(a)),S)))/(1+(np.dot(np.dot(S,a),a)))
            x = x + (np.dot(S,np.dot(np.matrix(a).transpose(),(np.matrix(b)-np.dot(np.matrix(a),x)))))
        return x

    # Hybrid learning, i.e. Descent Gradient for precedents and Least Squares Estimation for consequents.
    # trainHybridJangOffLine(学習回数, 学習を終える誤差, )
    def trainHybridJangOffLine(self, epochs=5, tolerance=1e-5, initialGamma=1000, k=0.01):

        self.trainingType = 'trainHybridJangOffLine'
        convergence = False
        epoch = 1

        while (epoch < epochs) and (convergence is not True):

            #layer four: forward pass
            [layerFour, wSum, w] = forwardHalfPass(self, self.X)

            #layer five: least squares estimate
            layerFive = np.array(self.LSE(layerFour,self.Y,initialGamma))
            self.consequents = layerFive
            layerFive = np.dot(layerFour,layerFive)

            #error
            error = np.sum((self.Y-layerFive.T)**2)
            print(str(epoch) + ' current error: ' + str(error))
            self.errors = np.append(self.errors,error)
            if error < self.min_error:
                self.min_error = error

            if len(self.errors) != 0:
                if self.errors[len(self.errors)-1] < tolerance:
                    convergence = True

            # back propagation
            if convergence is not True:
                cols = list(range(len(self.X[0,:])))
                dE_dAlpha = list(backprop(self, colX, cols, wSum, w, layerFive) for colX in range(self.X.shape[1]))


            if len(self.errors) >= 4:
                if (self.errors[-4] > self.errors[-3] > self.errors[-2] > self.errors[-1]):
                    k = k * 1.1

            if len(self.errors) >= 5:
                if (self.errors[-1] < self.errors[-2]) and (self.errors[-3] < self.errors[-2]) and (self.errors[-3] < self.errors[-4]) and (self.errors[-5] > self.errors[-4]):
                    k = k * 0.9

            ## handling of variables with a different number of MFs
            t = []
            for x in range(len(dE_dAlpha)):
                for y in range(len(dE_dAlpha[x])):
                    for z in range(len(dE_dAlpha[x][y])):
                        t.append(dE_dAlpha[x][y][z])

            eta = k / np.abs(np.sum(t))

            if(np.isinf(eta)):
                eta = k

            ## handling of variables with a different number of MFs
            dAlpha = copy.deepcopy(dE_dAlpha)
            if not(self.memFuncsHomo):
                for x in range(len(dE_dAlpha)):
                    for y in range(len(dE_dAlpha[x])):
                        for z in range(len(dE_dAlpha[x][y])):
                            dAlpha[x][y][z] = -eta * dE_dAlpha[x][y][z]
            else:
                dAlpha = -eta * np.array(dE_dAlpha)


            for varsWithMemFuncs in range(len(self.memFuncs)):
                for MFs in range(len(self.memFuncsByVariable[varsWithMemFuncs])):
                    paramList = sorted(self.memFuncs[varsWithMemFuncs][MFs][1])
                    for param in range(len(paramList)):
                        self.memFuncs[varsWithMemFuncs][MFs][1][paramList[param]] = self.memFuncs[varsWithMemFuncs][MFs][1][paramList[param]] + dAlpha[varsWithMemFuncs][MFs][param]
            epoch = epoch + 1


        self.fittedValues = predict(self,self.X)
        self.residuals = self.Y - self.fittedValues[:,0]

        return self.fittedValues


    def plotErrors(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(list(range(len(self.errors))),self.errors,'o', label='errors')
            plt.ylabel('error')
            plt.xlabel('epoch')
            plt.show()

    def plotMF(self, x, inputNumber):
        """plotMF(self, x, inputNumber)

        Parameters
        ----------
        x : 1d array or iterable
            表示したいmfの横軸の値
        inputNumber : int
            入力するデータの種類を配列の添字で示したもの

        """
        
        import matplotlib.pyplot as plt
        from skfuzzy import gaussmf, gbellmf, sigmf

        """gaussmf()

        Parameters
        ----------
        x : 1d array or iterable
            横軸の値
        mean : float
            中央値
        sigma : float
            標準偏差

        Returns
        -------
        y : 1d array
            関数の出力

        """

        for mf in range(len(self.memFuncs[inputNumber])):
            if self.memFuncs[inputNumber][mf][0] == 'gaussmf':
                y = gaussmf(x,**self.memClass.MFList[inputNumber][mf][1])
            elif self.memFuncs[inputNumber][mf][0] == 'gbellmf':
                y = gbellmf(x,**self.memClass.MFList[inputNumber][mf][1])
            elif self.memFuncs[inputNumber][mf][0] == 'sigmf':
                y = sigmf(x,**self.memClass.MFList[inputNumber][mf][1])

            plt.plot(x,y,'r')

        plt.show()

    def plotResults(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(map(plusOne, list(range(len(self.fittedValues)))),self.fittedValues,'or', label='trained')
            plt.plot(map(plusOne, list(range(len(self.Y)))),self.Y,'^b', label='original')
            plt.hlines([0.5], 0, len(self.fittedValues) + 1, "black", linestyles='dashed')
            plt.legend(loc='upper left')
            plt.show()

    def getAccuracyRate(self):
        predicted_data = self.fittedValues > 0.5
        correct_data = self.Y > 0.5

        cnt = 0
        for x, y in zip(predicted_data, correct_data):
            if x[0] == y:
                cnt+=1
        
        return float(cnt) / len(correct_data)

def plusOne(n):
    return n + 1

# Xsには，121組のmeanとsigmaが配列形式で格納されている
# 第1層から第4層まで
# return 第4層の出力, 適応率の合計, 適応率の配列？
def forwardHalfPass(ANFISObj, Xs):
    layerFour = np.empty(0,)
    wSum = []

    # なんのpatternだろう？
    for pattern in range(len(Xs[:,0])):
        # layer one: 入力をメンバシップ関数に適用する
        # layerOne: [[val, val, val, val], [val, val, val, val], ...]
        # 単純にMFListのすべてに値を適用した結果を返す
        layerOne = ANFISObj.memClass.evaluateMF(Xs[pattern,:])

        # layer two
        # len(ANFISObj.rules): 各メンバシップ関数の中身の直積 4^メンバシップ関数の数
        # len(ANFISObj.rules[0]): メンバシップ関数の中身 常に4
        # for row in range(len(ANFISObj.rules)):
        #     for x in range(len(ANFISObj.rules[0])):
        #         layerOne[x][ANFISObj.rules[row][x]]
        # 4*x個しかない値を4^x個の要素をもつ配列に展開する．なぜ？
        miAlloc = [[layerOne[x][ANFISObj.rules[row][x]] for x in range(len(ANFISObj.rules[0]))] for row in range(len(ANFISObj.rules))]
        # np.product(x) は配列xの全要素の積．たとえば，np.product([1, 2, 3, 4]) ==> 24
        # miAllocでメンバシップ関数の出力を展開したのは，一発ですべての積を計算するためだった
        # .T は転置
        # layerTwo: メンバシップ関数の値を掛け合わせた値である「適応度」を計算した
        # layerTwo ==> [w1, w2, w3, ... , w4^x] の1次元配列
        layerTwo = np.array([np.product(x) for x in miAlloc]).T
        if pattern == 0:
            w = layerTwo
        else:
            w = np.vstack((w,layerTwo))

        # layer three: 正規化
        wSum.append(np.sum(layerTwo))
        # numpyでは，[2, 4]/2 ==> [1, 2] となる
        # 4^x個の要素すべてを正規化する 形は変わらず [w1, w2, w3, ..., w4^x]            
        layerThree = layerTwo/wSum[pattern]

        # prep for layer four (bit of a hack)
        # np.append(Xs[pattern,:]) ==> [x1, x2, x3]
        # np.append(Xs[pattern,:],1) ==> [x1, x2, x3, 1]
        # for x in layerThree:
        #   x * np.append(Xs[pattern,:],1) ==> [x*x1, x*x2, x*x3, x*1]
        # 論文の w_bar * x_j にあたる 末尾に追加した1は，Σの外にある定数項の分
        rowHolder = np.concatenate([x*np.append(Xs[pattern,:],1) for x in layerThree])
        layerFour = np.append(layerFour,rowHolder)

    w = w.T
    layerFour = np.array(np.array_split(layerFour,pattern + 1))

    return layerFour, wSum, w


def backprop(ANFISObj, columnX, columns, theWSum, theW, theLayerFive):

    paramGrp = [0]* len(ANFISObj.memFuncs[columnX])
    for MF in range(len(ANFISObj.memFuncs[columnX])):

        parameters = np.empty(len(ANFISObj.memFuncs[columnX][MF][1]))
        timesThru = 0
        for alpha in sorted(ANFISObj.memFuncs[columnX][MF][1].keys()):

            bucket3 = np.empty(len(ANFISObj.X))
            for rowX in range(len(ANFISObj.X)):
                varToTest = ANFISObj.X[rowX,columnX]
                tmpRow = np.empty(len(ANFISObj.memFuncs))
                tmpRow.fill(varToTest)

                bucket2 = np.empty(ANFISObj.Y.ndim)
                for colY in range(ANFISObj.Y.ndim):

                    rulesWithAlpha = np.array(np.where(ANFISObj.rules[:,columnX]==MF))[0]
                    adjCols = np.delete(columns,columnX)

                    senSit = mfDerivs.partial_dMF(ANFISObj.X[rowX,columnX],ANFISObj.memFuncs[columnX][MF],alpha)
                    # produces d_ruleOutput/d_parameterWithinMF
                    dW_dAplha = senSit * np.array([np.prod([ANFISObj.memClass.evaluateMF(tmpRow)[c][ANFISObj.rules[r][c]] for c in adjCols]) for r in rulesWithAlpha])

                    bucket1 = np.empty(len(ANFISObj.rules[:,0]))
                    for consequent in range(len(ANFISObj.rules[:,0])):
                        fConsequent = np.dot(np.append(ANFISObj.X[rowX,:],1.),ANFISObj.consequents[((ANFISObj.X.shape[1] + 1) * consequent):(((ANFISObj.X.shape[1] + 1) * consequent) + (ANFISObj.X.shape[1] + 1)),colY])
                        acum = 0
                        if consequent in rulesWithAlpha:
                            acum = dW_dAplha[np.where(rulesWithAlpha==consequent)] * theWSum[rowX]

                        acum = acum - theW[consequent,rowX] * np.sum(dW_dAplha)
                        acum = acum / theWSum[rowX]**2
                        bucket1[consequent] = fConsequent * acum

                    sum1 = np.sum(bucket1)

                    if ANFISObj.Y.ndim == 1:
                        bucket2[colY] = sum1 * (ANFISObj.Y[rowX]-theLayerFive[rowX,colY])*(-2)
                    else:
                        bucket2[colY] = sum1 * (ANFISObj.Y[rowX,colY]-theLayerFive[rowX,colY])*(-2)

                sum2 = np.sum(bucket2)
                bucket3[rowX] = sum2

            sum3 = np.sum(bucket3)
            parameters[timesThru] = sum3
            timesThru = timesThru + 1

        paramGrp[MF] = parameters

    return paramGrp


def predict(ANFISObj, varsToTest):

    [layerFour, wSum, w] = forwardHalfPass(ANFISObj, varsToTest)

    #layer five
    layerFive = np.dot(layerFour,ANFISObj.consequents)

    return layerFive

if __name__ == "__main__":
    print("I am main!")