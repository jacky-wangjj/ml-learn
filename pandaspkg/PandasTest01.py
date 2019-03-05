#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : PandasTest01
# @Time     : 2019/3/5
# ------------------------
import pandas as pd
import numpy as np
# pandas基本数据结构有两个：series、dataframe

class PandasTest01:
    def __init__(self):
        pass

    def test(self):
        # 创建Series
        obj = pd.Series([1, 2, 3, 4])
        print('series:\n', obj[1], obj.values, obj.index)
        d = [[1, 2, 3, 4], [5, 6, 7, 8]]
        index = ['one', 'two']
        # 创建DataFrame
        df = pd.DataFrame(d, index = index)
        # loc通过行标签索引来确定行
        print('df.loc:\n', df.loc['one'])
        # iloc通过行号索引来确定行
        print('df.iloc:\n', df.iloc[0])
        # 读取列
        print('df[0]:\n', df[0])
        print('df.loc[:,[0]]:\n', df.loc[:,[0]])
        print('df.iloc[:,[0]]:\n', df.iloc[:,[0]])
        # 转为np.array后取一列
        print('np.array(df.values)[:,0]:\n', np.array(df.values)[:,0])
        # 转为np.array后取一行
        print('np.array(df.values)[0,:]:\n', np.array(df.values)[0,:])

if __name__ == "__main__":
    c = PandasTest01()
    c.test()