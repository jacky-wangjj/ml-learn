#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : FileUtils
# @Time     : 2018/12/24
# ------------------------
import os


class FileUtils:
    def __init__(self):
        pass

    def base(self):
        cwd = os.getcwd()
        print "path:", cwd
        files = os.listdir(cwd)
        for file in files:
            print file,

    def test(self):
        file = 'test.txt'
        mode = 'w+'
        if not os.path.exists(file):
            fo = open(file, mode)
            print fo.name, fo.mode, fo.closed
            fo.write("hello, file")
            print fo.tell()
            fo.seek(0, 0)
            str = fo.readline()
            print str
            fo.close()
            os.remove(file)
        return;


if __name__ == "__main__":
    c = FileUtils()
    c.base()
    c.test()