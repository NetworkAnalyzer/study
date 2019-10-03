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
import time

class ANFIS:
    def __init__(self, trainX, trainY, testX, testY, mf):
        self.trainX = np.array(copy.copy(trainX))
        self.trainY = np.array(copy.copy(trainY))
        self.testX = np.array(copy.copy(testX))
        self.testY = np.array(copy.copy(testY))

        self.memClass = copy.deepcopy(mf)
        # self.memFuncs = [
        #     [
        #         ['gaussmf', {'mean': 0.5, 'sigma': 10.0}],
        #         ['gaussmf', {'mean': 0.5, 'sigma': 7.0}]
        #     ],
        #     [
        #         ['gaussmf', {'mean': 0.5, 'sigma': 5.0}],
        #         ['gaussmf', {'mean': 10, 'sigma': 5.0}]
        #     ]
        # ]
        self.memFuncs = self.memClass.MFList
        # self.memFuncsByVariable = [
        #     [0, 1],
        #     [0, 1]
        # ]
        self.memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]
        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))
        self.consequents = np.empty(self.trainY.ndim * len(self.rules) * (self.trainX.shape[1] + 1))
        self.consequents.fill(0)
        self.errors = np.empty(0)
        self.min_error = 100
        # homo: 同じ
        # self.memFuncsHomo = True | False
        self.memFuncsHomo = all(len(i)==len(self.memFuncsByVariable[0]) for i in self.memFuncsByVariable)
        self.isTrained = False

        """
        True Positive: 正解であるはずのデータを正解であると判定した
        False Negative: 不正解であるはずのデータを不正解であると判定した
        True Negative: 正解であるはずのデータを不正解であると判定した
        False Positive: 不正解であるはずのデータを正解であると判定した
        Accuracy: 正解・不正解にかかわらず，正しい判定をした割合 ((TP + FN) / (TP + FN + TN + FP))
        Presicion: 正解と判定したものにおいて，正しい判定をした割合 (TP / (TP + FP))
        Recall: 正解と判定されるべきものにおいて，正しい判定をした割合 (TP / (TP + TN))
        """
        self.TP = 0
        self.FN = 0
        self.TN = 0
        self.FP = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_measure = 0

        self.time = 0

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

    def train(self, epochs=5, tolerance=1e-5, initialGamma=1000, k=0.01):
        # Jang's Hybrid off-line training
        start = time.time()

        convergence = False
        epoch = 1

        while (epoch < epochs) and (convergence is not True):

            #layer four: forward pass
            [layerFour, wSum, w] = forwardHalfPass(self, self.trainX)

            #layer five: least squares estimate
            layerFive = np.array(self.LSE(layerFour,self.trainY,initialGamma))
            self.consequents = layerFive
            # np.dot: 内積
            layerFive = np.dot(layerFour,layerFive)

            #error
            error = np.sum((self.trainY-layerFive.T)**2)
            print(str(epoch) + ' current error: ' + str(error))
            self.errors = np.append(self.errors,error)
            if error < self.min_error:
                self.min_error = error

            if error < tolerance:
                convergence = True

            # back propagation
            if convergence is not True:
                cols = list(range(len(self.trainX[0,:])))
                # dE_dAlpha = [
                #     [array([a, b]), array([c, d])],
                #     [array([e, f]), array([g, h])]
                # ]
                dE_dAlpha = list(backprop(self, colX, cols, wSum, w, layerFive) for colX in range(self.trainX.shape[1]))


            if len(self.errors) >= 4:
                if (self.errors[-4] > self.errors[-3] > self.errors[-2] > self.errors[-1]):
                    k = k * 1.1

            if len(self.errors) >= 5:
                if (self.errors[-1] < self.errors[-2]) and (self.errors[-3] < self.errors[-2]) and (self.errors[-3] < self.errors[-4]) and (self.errors[-5] > self.errors[-4]):
                    k = k * 0.9

            # t = [a b c d e f g h]
            t = []
            for x in range(len(dE_dAlpha)):
                for y in range(len(dE_dAlpha[x])):
                    for z in range(len(dE_dAlpha[x][y])):
                        t.append(dE_dAlpha[x][y][z])

            # eta: 学習率
            eta = k / np.abs(np.sum(t))

            # isinf: 無限大かどうか
            # tの各値がe-17くらいのとき，sum(t)が0になる
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
                # dAlpha = [
                #     [[-eta*a -eta*b][-eta*c -eta*d]]
                #     [[-eta*e -eta*f][-eta*g -eta*h]]
                # ]
                dAlpha = -eta * np.array(dE_dAlpha)


            for i in range(len(self.memFuncs)):
                for MFs in range(len(self.memFuncsByVariable[i])):
                    # paramList = ['mean', 'sigma']
                    paramList = sorted(self.memFuncs[i][MFs][1])
                    for param in range(len(paramList)):
                        # Update memFuncs
                        self.memFuncs[i][MFs][1][paramList[param]] += dAlpha[i][MFs][param]
            
            epoch = epoch + 1

        self.fittedValues = predict(self, self.testX)
        self.residuals = self.testY - self.fittedValues

        self.aggregate()

        end = time.time()
        self.time = end - start

        self.isTrained = True


    def plotErrors(self):
        if self.isTrained:
            import matplotlib.pyplot as plt
            plt.plot(list(range(len(self.errors))),self.errors,'o', label='errors')
            plt.ylabel('error')
            plt.xlabel('epoch')
            plt.show()
        else:
            print('ANFIS is not trained yet.')

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
        if self.isTrained:
            import matplotlib.pyplot as plt
            plt.plot(list(map(plusOne, list(range(len(self.fittedValues))))), self.fittedValues, 'or', label='trained')
            plt.plot(list(map(plusOne, list(range(len(self.testY))))), self.testY, '^b', label='original')
            plt.hlines([0.5], 0, len(self.fittedValues) + 1, "black", linestyles='dashed')
            plt.legend(loc='upper left')
            plt.show()
        else:
            print('ANFIS is not trained yet.')

    def aggregate(self):
        self.TP = self.FN = self.TN = self.FP = 0
        self.accuracy = self.precision = self.recall = 0

        predicted_data = [x > 0.5 for x in self.fittedValues]
        correct_data = self.testY > 0.5

        for x, y in zip(predicted_data, correct_data):
            if y == 1:
                if x == 1:
                    self.TP+=1
                elif x == 0:
                    self.TN+=1
            elif y == 0:
                if x == 1:
                    self.FP+=1
                elif x == 0:
                    self.FN+=1

        self.accuracy = float(self.TP + self.FN) / (self.TP + self.FN + self.TN + self.FP)
        self.precision = float(self.TP) / (self.TP + self.FP) if self.TP + self.FP != 0 else 0 
        self.recall = float(self.TP) / (self.TP + self.TN) if self.TP + self.TN != 0 else 0
        self.f_measure = 2 * self.precision * self.recall / (self.precision + self.recall) if self.precision + self.recall != 0 else 0

def plusOne(n):
    return n + 1

def forwardHalfPass(anfis, trainData):
    layerFour = np.empty(0,)
    wSum = []

    for i in range(len(trainData[:,0])):
        # layer 1: 入力をメンバシップ関数に適用する
        #          1つの入力に対して複数のMFがあるので値が増える
        # trainData[i,:] => [0.0589 0.6108]
        # layerOne => [[0.9990, 0.9980], [0.9570, 0.9843]]
        layerOne = anfis.memClass.evaluateMF(trainData[i,:])

        # layer 2: 適応度 (MFの出力を掛け合わせた値) を計算する
        # layerOne => [[a1, a2], [b1, b2]] のとき
        # miAlloc  => [[a1, b1], [a1, b2], [a2, b1], [a2, b2]] (layerOne[0]とlayerOne[1]の直積)
        # layerTwo => [a1*b1 a1*b2 a2*b1 a2*b2]
        miAlloc = [[layerOne[x][anfis.rules[row][x]] for x in range(len(anfis.rules[0]))] for row in range(len(anfis.rules))]
        layerTwo = np.array([np.product(x) for x in miAlloc]).T

        # 適応度を重みとして扱う
        # w = [w1 w2 w3 w4]
        w = layerTwo if i == 0 else np.vstack((w,layerTwo))

        # layer 3: 適応度を正規化する
        # layerThree = [w1/wSum[i] w2/wSum[i] w3/wSum[i] w4/wSum[i]]
        wSum.append(np.sum(layerTwo))
        layerThree = layerTwo/wSum[i]

        # layer 4: 何をしているかわからない
        # np.append(trainData[i,:],1) = [a b 1]
        # [x * [a b 1] for x in [w1 w2 w3 w4]] = [
        #     array[w1*a w1*b w1*1],
        #     array[w2*a w2*b w2*1],
        #     array[w3*a w3*b w3*1],
        #     array[w4*a w4*b w4*1]
        # ]
        # rowHolder = [
        #    w1*a w1*b w1*1 w2*a w2*b w2*1
        #    w3*a w3*b w3*1 w4*a w4*b w4*1
        # ]
        rowHolder = np.concatenate([x*np.append(trainData[i,:],1) for x in layerThree])
        layerFour = np.append(layerFour,rowHolder)

    w = w.T
    layerFour = np.array(np.array_split(layerFour,i + 1))

    return layerFour, wSum, w


def backprop(ANFISObj, columnX, columns, theWSum, theW, theLayerFive):

    paramGrp = [0]* len(ANFISObj.memFuncs[columnX])
    for MF in range(len(ANFISObj.memFuncs[columnX])):

        parameters = np.empty(len(ANFISObj.memFuncs[columnX][MF][1]))
        timesThru = 0
        for alpha in sorted(ANFISObj.memFuncs[columnX][MF][1].keys()):

            bucket3 = np.empty(len(ANFISObj.trainX))
            for rowX in range(len(ANFISObj.trainX)):
                varToTest = ANFISObj.trainX[rowX,columnX]
                tmpRow = np.empty(len(ANFISObj.memFuncs))
                tmpRow.fill(varToTest)

                bucket2 = np.empty(ANFISObj.trainY.ndim)
                for colY in range(ANFISObj.trainY.ndim):

                    rulesWithAlpha = np.array(np.where(ANFISObj.rules[:,columnX]==MF))[0]
                    adjCols = np.delete(columns,columnX)

                    senSit = mfDerivs.partial_dMF(ANFISObj.trainX[rowX,columnX],ANFISObj.memFuncs[columnX][MF],alpha)
                    # produces d_ruleOutput/d_parameterWithinMF
                    dW_dAplha = senSit * np.array([np.prod([ANFISObj.memClass.evaluateMF(tmpRow)[c][ANFISObj.rules[r][c]] for c in adjCols]) for r in rulesWithAlpha])

                    bucket1 = np.empty(len(ANFISObj.rules[:,0]))
                    for consequent in range(len(ANFISObj.rules[:,0])):
                        fConsequent = np.dot(np.append(ANFISObj.trainX[rowX,:],1.),ANFISObj.consequents[((ANFISObj.trainX.shape[1] + 1) * consequent):(((ANFISObj.trainX.shape[1] + 1) * consequent) + (ANFISObj.trainX.shape[1] + 1)),colY])
                        acum = 0
                        if consequent in rulesWithAlpha:
                            acum = dW_dAplha[np.where(rulesWithAlpha==consequent)] * theWSum[rowX]

                        acum = acum - theW[consequent,rowX] * np.sum(dW_dAplha)
                        acum = acum / theWSum[rowX]**2
                        bucket1[consequent] = fConsequent * acum

                    sum1 = np.sum(bucket1)

                    if ANFISObj.trainY.ndim == 1:
                        bucket2[colY] = sum1 * (ANFISObj.trainY[rowX]-theLayerFive[rowX,colY])*(-2)
                    else:
                        bucket2[colY] = sum1 * (ANFISObj.trainY[rowX,colY]-theLayerFive[rowX,colY])*(-2)

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