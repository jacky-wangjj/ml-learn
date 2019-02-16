#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------
# Author:   wangjj17
# Name:     TensorflowTest01
# Date:     2019/2/16
# -------------------------
import tensorflow as tf

class TensorflowTest01:
    def __init__(self):
        pass

    def test(self):
        hello = tf.constant('Hello TensorFlow!')
        sess = tf.Session()
        print(sess.run(hello))

if __name__ == "__main__":
    c = TensorflowTest01()
    c.test()