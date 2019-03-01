#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : SklearnTest02
# @Time     : 2019/2/28
# ------------------------
import numpy as np
from sklearn import linear_model, svm, neighbors, ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt

class SklearnTest02:
    def __init__(self):
        pass

    def f(x1, x2):
        y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2) + 0.1 * x1 + 3
        return y

    def load_data(self):
        x1_train = np.linspace(0, 50, 500)
        x2_train = np.linspace(-10, 10, 500)
        data_train = np.array([[x1, x2, SklearnTest02.f(x1, x2) + (np.random.random(1) - 0.5)] for x1, x2 in zip(x1_train, x2_train)])
        x1_test = np.linspace(0, 50, 100) + 0.5 * np.random.random(100)
        x2_test = np.linspace(-10, 10, 100) + 0.02 * np.random.random(100)
        data_test = np.array([[x1, x2, SklearnTest02.f(x1, x2)] for x1, x2 in zip(x1_test, x2_test)])
        return data_train, data_test

    def get_data(self):
        data_train, data_test = SklearnTest02().load_data()
        x_train = np.mat(data_train[:, [0, 1]])
        y_train = np.mat(data_train[:, 2]).T
        # print(data_train)
        # print(x_train)
        # print(y_train)
        x_test = np.mat(data_test[:, [0, 1]])
        y_test = np.mat(data_test[:, 2]).T
        return x_train, y_train, x_test, y_test

    def try_different_method(self, clf, x_train, y_train, x_test, y_test):
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        y_pred = clf.predict(x_test)
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
        # the plot
        plt.figure()
        plt.plot(np.arange(len(y_pred)), y_test, 'go-', label='true value')
        plt.plot(np.arange(len(y_pred)), y_pred, 'ro-', label='predict value')
        plt.title('score:%f' % score)
        plt.legend()
        plt.show()

    # 回归树
    def DecisionTreeRegressTest(self):
        x_train, y_train, x_test, y_test = SklearnTest02().get_data()
        clf = DecisionTreeRegressor()
        SklearnTest02().try_different_method(clf, x_train, y_train, x_test, y_test)

    # 线性回归
    def LinearRegressionTest(self):
        x_train, y_train, x_test, y_test = SklearnTest02().get_data()
        linear_reg = linear_model.LinearRegression()
        SklearnTest02().try_different_method(linear_reg, x_train, y_train, x_test, y_test)

    # SVM支持向量机
    def SVMTest(self):
        x_train, y_train, x_test, y_test = SklearnTest02().get_data()
        svr = svm.SVR()
        SklearnTest02().try_different_method(svr, x_train, y_train, x_test, y_test)

    # Kmeans
    def KNeighborsRegressorTest(self):
        x_train, y_train, x_test, y_test = SklearnTest02().get_data()
        knn = neighbors.KNeighborsRegressor()
        SklearnTest02().try_different_method(knn, x_train, y_train, x_test, y_test)

    # 随机森林
    def RandomForestRegressorTest(self):
        x_train, y_train, x_test, y_test = SklearnTest02().get_data()
        rf = ensemble.RandomForestRegressor(n_estimators=20)#使用20个决策树
        SklearnTest02().try_different_method(rf, x_train, y_train, x_test, y_test)

    # Adaboost
    def AdaboostTest(self):
        x_train, y_train, x_test, y_test = SklearnTest02().get_data()
        ada = ensemble.AdaBoostRegressor(n_estimators=50)
        SklearnTest02().try_different_method(ada, x_train, y_train, x_test, y_test)

    # GBRT
    def GradientBoostingRegressorTest(self):
        x_train, y_train, x_test, y_test = SklearnTest02().get_data()
        gbrt = ensemble.GradientBoostingRegressor(n_estimators=100)
        SklearnTest02().try_different_method(gbrt, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    c = SklearnTest02()
    # c.DecisionTreeRegressTest()
    # c.LinearRegressionTest()
    c.SVMTest()
    # c.KNeighborsRegressorTest()
    # c.RandomForestRegressorTest()
    # c.AdaboostTest()
    # c.GradientBoostingRegressorTest()
