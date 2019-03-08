#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------
# Author:   wangjj17
# Name:     PolynomialRegression
# Date:     2019/3/7
# -------------------------
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self):
        pass

    def test(self):
        X = np.arange(1, 11).reshape(-1, 2)
        print('X.shape:', X.shape)
        print('X:\n', X)
        # degree=2
        poly = PolynomialFeatures(degree=2)
        poly.fit(X)
        X2 = poly.transform(X)
        print('X2.shape:', X2.shape)
        print('X2:\n', X2)
        # degree=3
        poly = PolynomialFeatures(degree=3)
        X3 = poly.fit_transform(X)
        print('X3.shape:', X3.shape)
        print('X3:\n', X3)
        # 模拟数据生成
        x = np.random.uniform(-3, 3, size=100)
        XX = x.reshape(-1, 1)
        y = 0.5*x**2+x+2+np.random.normal(0, 1, 100)
        # 实例化pipeline
        poly_reg = Pipeline([
            ("poly", PolynomialFeatures(degree=2)),
            ("std_scaler", StandardScaler()),
            ("lin_reg", LinearRegression())
        ])
        poly_reg.fit(XX, y)
        y_predict = poly_reg.predict(XX)
        # 画出散点图和拟合曲线
        plt.scatter(x, y)
        plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
        plt.show()

    def test2(self):
        X_train = [[6], [8], [10], [14], [18]]
        y_train = [[7], [9], [13], [17.5], [18]]
        X_test = [[6], [8], [11], [16]]
        y_test = [[8], [12], [15], [18]]
        # 建立线性回归，并用训练的模型绘图
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        xx = np.linspace(0, 26, 100)
        yy = regressor.predict(xx.reshape(xx.shape[0], 1))
        plt.figure()
        plt.plot(X_train, y_train, 'k.')
        plt.plot(xx, yy)
        # 二次多项式拟合
        quadratic_featurizer = PolynomialFeatures(degree=2)
        X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
        X_test_quadratic = quadratic_featurizer.transform(X_test)
        regressor_quadratic = LinearRegression()
        regressor_quadratic.fit(X_train_quadratic, y_train)
        xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
        plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')
        # 三次多项式拟合
        cubic_featurizer = PolynomialFeatures(degree=3)
        X_train_cubic = cubic_featurizer.fit_transform(X_train)
        X_test_cubic = cubic_featurizer.transform(X_test)
        regressor_cubic = LinearRegression()
        regressor_cubic.fit(X_train_cubic, y_train)
        xx_cubic = cubic_featurizer.transform(xx.reshape(xx.shape[0], 1))
        plt.plot(xx, regressor_cubic.predict(xx_cubic))
        plt.show()
        print("xx:\n", xx)
        print("xx.reshape(xx.shape[0], 1):\n", xx.reshape(xx.shape[0], 1))

if __name__ == "__main__":
    c = PolynomialRegression()
    # c.test()
    c.test2()