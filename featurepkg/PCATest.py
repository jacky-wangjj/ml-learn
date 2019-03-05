#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : PCATest
# @Time     : 2019/3/5
# ------------------------
import numpy as np
from sklearn.decomposition import PCA

class PCATest:
    def __init__(self):
        pass

    def test(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        pca = PCA(n_components=2)
        pca.fit(X)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)
        pca1 = PCA(n_components=2, svd_solver='full')
        pca1.fit(X)
        print(pca1.explained_variance_ratio_)
        print(pca1.singular_values_)
        pca2 = PCA(n_components=1, svd_solver='arpack')
        pca2.fit(X)
        print(pca2.explained_variance_ratio_)
        print(pca2.singular_values_)

if __name__ == "__main__":
    c = PCATest()
    c.test()