#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : PlotTest
# @Time     : 2018/12/24
# ------------------------
import numpy as np
import matplotlib.pyplot as plt

class PlotTest:
    def __init__(self):
        pass

    def plot(self):
        # 线的绘制
        x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
        c, s = np.cos(x), np.sin(x)
        # 绘制
        plt.figure(1)
        # 自变量 因变量
        plt.plot(x, c)
        # 自变量 因变量
        plt.plot(x, s)
        plt.savefig("one.png")
        plt.show()

if __name__ == "__main__":
    c = PlotTest()
    c.plot()