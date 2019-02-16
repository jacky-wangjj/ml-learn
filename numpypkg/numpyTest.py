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
        print(np.eye(4))
        print(np.array([1, 2, 3]))
        print(np.array([[1, 2], [3, 4]]))
        print(np.array([1, 2, 3], dtype = complex))
        student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
        print(np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype = student))

    def numpyProps(self):
        a = np.arange(24)
        print(a.ndim, a)
        b = a.reshape(4, 3, 2)
        print(b.ndim, b)
        print(b.shape)
        x = np.array([1,2,3,4,5], dtype = np.int8) #数组dtype为int8(一个字节)
        print(x.itemsize)
        y = np.array([1,2,3,4,5], dtype = np.float64) #数组dtype为float64(八字节)
        print(y.itemsize)

    def createArray(self):
        x = (1, 2, 3)
        a = np.asarray(x, dtype=float)
        print(a)
        s = 'hello world'
        #b = np.frombuffer(s, dtype='S1')
        #print(b)
        list = range(5)
        it = iter(list)
        x = np.fromiter(it, dtype=float)
        print(x)
        c =np.logspace(1, 10, 10, base=2)
        print(c)
        ss = slice(2, 7, 2) #从索引2开始到索引7停止，间隔为2
        #print(b[ss])
        #print(b[2:7:2]) #从索引2开始到索引7停止，间隔为2
        #print(b[2:]) #从该索引开始以后的所有项
        d = np.array([[1,2,3],[3,4,5],[4,5,6]])
        print(d[...,1]) #第2列
        print(d[1,...]) #第2行
        print(d[...,1:]) #第2列及剩下的所有元素
        print(d[[0,1,2],[0,1,0]]) #获取(0,0),(1,1),(2,0)位置的元素

    def broadcast(self):
        a = np.array([[0, 0, 0],
                      [10, 10, 10],
                      [20, 20, 20],
                      [30, 30, 30]])
        b = np.array([1, 2, 3])
        print(a+b).T
        for x in np.nditer((a+b).T.copy(order='C')): #按行顺序排序
            print(x),
        for x in np.nditer((a+b).T.copy(order='F')): #按列顺序排序
            print(x),
        x = np.array([[1],[2],[3]])
        y = np.array([4, 5, 6])
        z = np.broadcast(x, y) #对y广播x
        print(z.shape)
        c = np.empty(z.shape)
        c.flat = [u + v for (u,v) in z]
        print(c)
        print(x+y)
        print(np.broadcast_to(y, (3,3)))
        print(np.broadcast_to(x, (3,3)))

    def concatenate(self):
        a = np.array([[1,2],[3,4]])
        b = np.array([[5,6],[7,8]])
        print(np.concatenate((a,b), axis = 0))
        print(np.concatenate((a,b), axis = 1))
        print('沿轴0堆叠两个数组：')
        print((np.stack((a,b),0)))
        print('沿轴1堆叠两个数组：')
        print(np.stack((a,b),1))
        print('沿轴2堆叠两个数组：')
        print(np.stack((a,b),2))
        print('水平堆叠：')
        print(np.hstack((a,b)))
        print('竖直堆叠：')
        print(np.vstack((a,b)))

if __name__ == "__main__":
    c = numpyTest()
    c.test()
    c.numpyProps()
    c.createArray()
    c.broadcast()
    c.concatenate()