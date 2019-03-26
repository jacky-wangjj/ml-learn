#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------
# Author:   wangjj17
# Name:     pretreatment
# Date:     2019/3/26
# -------------------------
import os
import pandas as pd
import numpy as np

path="D:\\Excel"
outFile="D:\\merge.csv"

def getFile(path):
    files = []
    pathDir = os.listdir(path)
    for allDir in pathDir:
        child = os.path.join('%s\\%s' % (path, allDir))
        if os.path.isfile(child) and os.path.splitext(child)[1].lower()=='.csv':
            files.append(child)
    print(files)
    return files

def read_csv(file):
    df = pd.read_csv(file, names=['wavelength','absorbance'])
    # print(df)
    return df

def merge_csv(path, outFile):
    files = getFile(path)
    out = []
    if os.path.exists(outFile):
        os.remove(outFile)
    for file in files:
        df = read_csv(file)
        if len(out) == 0:
            out.append(df['wavelength'])
        out.append(df['absorbance'])
    pd.DataFrame(out).to_csv(outFile, header=None)

if __name__ == "__main__":
    merge_csv(path, outFile)