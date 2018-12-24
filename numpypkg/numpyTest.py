#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : numpyTest
# @Time     : 2018/12/24
# ------------------------
import numpy as np

class numpyTest:
    def __init__(self):
        pass

    def test(self):
        print np.eye(4)
        print np.array([1, 2, 3])
        print np.array([[1, 2], [3, 4]])
        print np.array([1, 2, 3], dtype = complex)
        student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
        print np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype = student)

if __name__ == "__main__":
    c = numpyTest()