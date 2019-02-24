#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------
# Author:   wangjj17
# Name:     LeastSquares01
# Date:     2019/2/24
# -------------------------
# https://zhuanlan.zhihu.com/p/24042722
import numpy as np
import scipy as sp
import pylab as pl
from scipy.optimize import leastsq  #引入最小二乘函数

class LeastSquares01:
    def __init__(self):
        pass

    # 目标函数y=sin(2πx)
    def real_func(x):
        return np.sin(2 * np.pi * x)

    # 多项式函数(拟合函数，也就是h(x))
    def fit_func(p, x):
        f = np.poly1d(p)
        return f(x)

    # 残差函数
    def residuals_func(p, y, x):
        # 正则化系数lambda
        regularization = 0.00001
        ret = LeastSquares01.fit_func(p, x) - y
        # 将lambda^(1/2)p加在了返回的array的后面
        ret = np.append(ret, np.sqrt(regularization) * p)
        return ret

    def test(self):
        n = 9  # 多项式次数
        # 随机选取0-1之间的9个数作为x
        x = np.linspace(0, 1, 9)

        # 画图时需要的连续点
        x_points = np.linspace(0, 1, 10000)

        # 目标函数
        y0 = LeastSquares01.real_func(x)
        # 添加正太分布噪声后的函数
        y1 = [np.random.normal(0, 0.1) + y for y in y0]

        # 随机初始化多项式参数
        p_init = np.random.randn(n)

        plsq = leastsq(LeastSquares01.residuals_func, p_init, args=(y1, x))

        # 输出拟合参数
        print('Fitting Parameters: ', plsq[0])
        # Fitting Parameters:  [ -2.83923076e+02   1.14996949e+03  -2.06511905e+03   2.08842363e+03 -1.22138657e+03   4.03643606e+02  -8.37576650e+01   1.21696134e+01 8.60284229e-03]

        # 绘制y=sina(2πx)
        pl.plot(x_points, LeastSquares01.real_func(x_points), label='real', color="red")

        # 绘制拟合函数
        pl.plot(x_points, LeastSquares01.fit_func(plsq[0], x_points), label='fitted curve')

        # 绘制随机0-1之间的9个点
        pl.plot(x, y1, 'bo', label='with noise')

        pl.legend()
        pl.show()

if __name__ == "__main__":
    c = LeastSquares01()
    c.test()