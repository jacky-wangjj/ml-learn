#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------
# Author:   wangjj17
# Name:     SklearnTest01
# Date:     2019/2/23
# -------------------------
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

class SklearnTest01:
    def __init__(self):
        pass

    def test(self):
        lr = linear_model.LinearRegression()
        boston = datasets.load_boston()
        y = boston.target
        predicted = cross_val_predict(lr, boston.data, y, cv=10)
        fig, ax = plt.subplots()
        ax.scatter(y, predicted, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()

if __name__ == "__main__":
    c = SklearnTest01()
    c.test()