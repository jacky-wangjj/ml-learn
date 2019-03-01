#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : NIRFit03
# @Time     : 2019/2/28
# ------------------------
import numpy as np
from sklearn import linear_model, svm, neighbors, ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

class NIRFit03:
    def __init__(self):
        pass

    # 第1列是浓度，2-1558列是不同波长的吸光度
    # 第1行是光谱数据的不同波长，2-11行是不同浓度的多组光谱数据
    def loadDataSet(fileName):
        numFeat = len(open(fileName).readline().split(',')) - 1
        print(numFeat)
        dataMat = [];
        labelMat = []
        fr = open(fileName)
        lineNum = 0
        for line in fr.readlines():
            lineArr = []
            lineNum += 1
            if (lineNum != 1):
                curLine = line.strip().split(',')
                for i in range(1, numFeat + 1):
                    lineArr.append(float(curLine[i]))
                dataMat.append(lineArr)
                labelMat.append(float(curLine[0]))
        return dataMat, labelMat

    def get_data(self):
        xArr, yArr = NIRFit03.loadDataSet("dataset2.CSV")
        # 画出数据集光谱各波长对应的吸光度
        NIRFit03().drawNIR(xArr, yArr)
        # PCA降维，由入参指定降到的维数
        red_X = NIRFit03().pca_op(xArr, 200)
        NIRFit03().drawNIR(red_X, yArr)
        # xMat = np.mat(xArr);
        xMat = np.mat(red_X);
        yMat = np.mat(yArr).T
        # 拆分集合为训练集合测试集
        x_train, x_test, y_train, y_test = train_test_split(xMat, yMat, test_size=0.2)
        print('X train set:\n', x_train)
        print('X test set:\n', x_test)
        print('Y train set:\n', y_train)
        print('Y test set:\n', y_test)
        return x_train, y_train, x_test, y_test

    # 画出数据集光谱各波长对应的吸光度
    def drawNIR(self, xArr, yArr):
        plt.figure()
        for i in range(len(yArr)):
            plt.plot(np.arange(len(xArr[i])), xArr[i], label=yArr[i])
        plt.title('NIR data')
        plt.legend()
        plt.show()

    # 回归系数的分布直方图，可查看是否能降维
    def drawCoef(self, coef):
        f = plt.figure(figsize=(7, 5))
        ax = f.add_subplot(111)
        ax.hist(coef.T, bins=50, color='b')
        ax.set_title("Histogram of the regr.coef_")
        plt.show()

    # 画出测试集预测Y与真实Y的差距
    def drawPred(self, y_pred, y_test, score):
        # the plot
        plt.figure()
        plt.plot(np.arange(len(y_pred)), y_test, 'go-', label='true value')
        plt.plot(np.arange(len(y_pred)), y_pred, 'ro-', label='predict value')
        plt.title('score:%f' % score)
        plt.legend()
        plt.show()

    def pca_op(self, xArr, n):
        pca = PCA(n_components=n, whiten=False, svd_solver='auto')
        pca.fit(xArr)
        red_X = pca.transform(xArr)
        print("降维后的特征数:", pca.n_components_)
        return red_X

    # 测试各种回归算法，输出相关信息
    def try_different_method(self, clf, x_train, y_train, x_test, y_test):
        model = clf.fit(x_train, y_train)
        print('model: \n', model)
        score = clf.score(x_test, y_test)
        y_pred = clf.predict(x_test)
        print('predict Y: \n', y_pred)
        # The coefficients
        coef = clf.coef_
        print('Coefficients: \n', coef)
        NIRFit03().drawCoef(coef)
        # The Intercepts
        intercept = clf.intercept_
        print('Intercepts: \n', intercept)
        # score
        print("score: %.4f" % clf.score(x_test, y_test))
        # The mean squared error, 均方差
        print("Mean squared error: %.4f" % mean_squared_error(y_test, y_pred))
        # The mean absolute error, 平均绝对误差
        print('Mean absolute error: %.4f' % mean_absolute_error(y_test, y_pred))
        # The median absolute error, 中值绝对误差
        print('Median absoulte error: %.4f' % median_absolute_error(y_test, y_pred))
        # Explained variance score: 1 is perfect prediction; r2->0模型越差，r2->1模型越好
        # R2决定系数，表征回归方程在多大程度上解释了因变量的变化，或者说方程对观测值的拟合程度如何。
        print('Variance score: %.4f' % r2_score(y_test, y_pred))
        # The cross validation
        X = np.vstack((x_train, x_test))
        Y = np.vstack((y_train, y_test))
        scores = cross_val_score(clf, X, Y, cv=int(len(Y)*0.8))
        print("Cross val score: \n", scores)
        print("Mean score: %.4f" % scores.mean())
        # The cross predict
        print("Cross val predicte:\n", cross_val_predict(clf, X, Y, cv=int(len(Y)*0.9)))
        # the plot
        NIRFit03().drawPred(y_pred, y_test, score)

    # 回归树
    def DecisionTreeRegressTest(self):
        x_train, y_train, x_test, y_test = NIRFit03().get_data()
        clf = DecisionTreeRegressor()
        NIRFit03().try_different_method(clf, x_train, y_train, x_test, y_test)

    # 线性回归
    def LinearRegressionTest(self):
        x_train, y_train, x_test, y_test = NIRFit03().get_data()
        linear_reg = linear_model.LinearRegression()
        NIRFit03().try_different_method(linear_reg, x_train, y_train, x_test, y_test)

    # SVM支持向量机
    def SVMTest(self):
        x_train, y_train, x_test, y_test = NIRFit03().get_data()
        svr = svm.SVR()
        NIRFit03().try_different_method(svr, x_train, y_train, x_test, y_test)

    # Kmeans
    def KNeighborsRegressorTest(self):
        x_train, y_train, x_test, y_test = NIRFit03().get_data()
        knn = neighbors.KNeighborsRegressor()
        NIRFit03().try_different_method(knn, x_train, y_train, x_test, y_test)

    # 随机森林
    def RandomForestRegressorTest(self):
        x_train, y_train, x_test, y_test = NIRFit03().get_data()
        rf = ensemble.RandomForestRegressor(n_estimators=20)#使用20个决策树
        NIRFit03().try_different_method(rf, x_train, y_train, x_test, y_test)

    # Adaboost
    def AdaboostTest(self):
        x_train, y_train, x_test, y_test = NIRFit03().get_data()
        ada = ensemble.AdaBoostRegressor(n_estimators=50)
        NIRFit03().try_different_method(ada, x_train, y_train, x_test, y_test)

    # GBRT
    def GradientBoostingRegressorTest(self):
        x_train, y_train, x_test, y_test = NIRFit03().get_data()
        gbrt = ensemble.GradientBoostingRegressor(n_estimators=100)
        NIRFit03().try_different_method(gbrt, x_train, y_train, x_test, y_test)

    # 多项式回归
    def PolynominalRegressorTest(self):
        x_train, y_train, x_test, y_test = NIRFit03().get_data()
        featurizer = PolynomialFeatures(degree=2) #degree定义是最高次项
        x_train_featurizer = featurizer.fit_transform(x_train) #fit_transform：正则化
        x_test_featurizer = featurizer.fit_transform(x_test) #fit_transform：正则化
        lr = linear_model.LinearRegression()
        NIRFit03().try_different_method(lr, x_train_featurizer, y_train, x_test_featurizer, y_test)

if __name__ == "__main__":
    c = NIRFit03()
    # c.DecisionTreeRegressTest()
    c.LinearRegressionTest()
    # c.SVMTest()
    # c.KNeighborsRegressorTest()
    # c.RandomForestRegressorTest()
    # c.AdaboostTest()
    # c.GradientBoostingRegressorTest()
    # c.PolynominalRegressorTest()