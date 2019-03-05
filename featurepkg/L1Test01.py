#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : L1Test01
# @Time     : 2019/3/5
# ------------------------
# Scikit-Learn库逻辑斯蒂L1正则化-特征选择
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class L1Test01:
    def __init__(self):
        pass

    def test(self):
        # 导入数据
        df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
        df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                           'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
                           'Hue', 'OD280/OD315 of diluted wines', 'Proline']
        print('class labels:', np.unique(df_wine['Class label']))
        # print (df_wine.head(5))
        # 分割训练集合测试集
        X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        # 特征值缩放
        # 归一化
        mms = MinMaxScaler()
        X_train_norm = mms.fit_transform(X_train)
        X_test_norm = mms.fit_transform(X_test)
        # 标准化
        stdsc = StandardScaler()
        X_train_std = stdsc.fit_transform(X_train)
        X_test_std = stdsc.fit_transform(X_test)

        # L1正则化的逻辑斯蒂模型
        lr = LogisticRegression(penalty='l1', C=0.1)  # penalty='l2'
        lr.fit(X_train_std, y_train)
        print('Training accuracy:', lr.score(X_train_std, y_train))
        print('Test accuracy:', lr.score(X_test_std, y_test))  # 比较训练集和测试集，观察是否出现过拟合
        print(lr.intercept_)  # 查看截距，三个类别
        print(lr.coef_)  # 查看权重系数，L1有稀疏化效果做特征选择

        # 正则化效果，减少约束参数值C，增加惩罚力度，各特征权重系数趋近于0
        fig = plt.figure()
        ax = plt.subplot(111)
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue',
                  'gray', 'indigo', 'orange']
        weights, params = [], []
        for c in np.arange(-4, 6, dtype=float):
            lr = LogisticRegression(penalty='l1', C=10 ** c, random_state=0)
            lr.fit(X_train_std, y_train)
            weights.append(lr.coef_[0])  # 三个类别，选择第一个类别来观察
            params.append(10 ** c)
        weights = np.array(weights)
        for column, color in zip(range(weights.shape[1]), colors):
            plt.plot(params, weights[:, column], label=df_wine.columns[column + 1], color=color)
        plt.axhline(0, color='black', linestyle='--', linewidth=3)
        plt.xlim([10 ** (-5), 10 ** 5])
        plt.ylabel('weight coefficient')
        plt.xlabel('C')
        plt.xscale('log')
        plt.legend(loc='upper left')
        ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
        plt.show()

if __name__ == "__main__":
    c = L1Test01()
    c.test()