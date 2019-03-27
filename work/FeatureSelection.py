#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------
# Author:   wangjj17
# Name:     FeatureSelection
# Date:     2019/3/21
# -------------------------
from sklearn.feature_selection import VarianceThreshold

# 移除低方差的特征
# threshold为方差阈值，低于方差阈值的特征都会被丢弃
def variance_select(X, threshold):
    sel = VarianceThreshold(threshold=threshold)
    X_sel = sel.fit_transform(X)
    return X_sel


