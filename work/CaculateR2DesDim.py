#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------
# Author:   wangjj17
# Name:     CaculateR2DesDim
# Date:     2019/3/26
# -------------------------
import numpy as np

# clf为模型
# step为计算R2的维度
# num为保留R2最高多少个step
def caculateR2(x_train, y_train, x_test, y_test, clf, step, num):
    m,n = x_train.shape
    print("x_train:\n", x_train)
    print("x_test:\n", x_test)
    scores = []
    if n%step == 0:
        count = n//step
    else:
        count = n//step + 1
    print("count:", count)
    for i in range(count):
        s_train = []
        s_test = []
        for j in range(step):
            # 超过维度n的最后一组使用最后一个维度重复填充
            if i*step+j >= n:
                s_train.append(x_train[:,n-1])
                s_test.append(x_test[:,n-1])
            else:
                s_train.append(x_train[:,i*step+j])
                s_test.append(x_test[:,i*step+j])
        s_train = np.mat(s_train).T
        s_test = np.mat(s_test).T
        # print("s_train:\n", s_train)
        # print("s_test:\n", s_test)
        # 使用s_X和yArr拟合clf，并计算R2
        clf.fit(s_train, y_train)
        scores.append(clf.score(s_test, y_test))
    # 得到scores降序索引
    s_index = np.argsort(-np.array(scores))
    print("s_scores:\n", np.sort(-np.array(scores)))
    print("s_index:", s_index)
    # 映射索引到真实索引
    index = []
    for k in range(num):
        for j in range(step):
            if s_index[k]*step+j == n:
                break
            else:
                index.append(s_index[k]*step+j)
    print("index:", index)
    x_train = x_train[:,index]
    x_test = x_test[:,index]
    return x_train, x_test
