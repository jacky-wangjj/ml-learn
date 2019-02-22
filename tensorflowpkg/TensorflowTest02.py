#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : TensorflowTest02
# @Time     : 2019/2/22
# ------------------------
import tensorflow as tf
import numpy as np
import matplotlib as plt

class TensorflowTest02:
    def __init__(self):
        pass

    def test(self):
        # 训练集X和对应的label Y，也就是零件的次品为0，正常零件为1
        X = [[0.7, 0.9],
             [0.1, 0.4],
             [0.5, 0.8],
             [0.6, 0.9],
             [0.2, 0.4],
             [0.6, 0.8]]
        # X = np.array(X)
        Y = [[1., 0., 1., 1., 0., 1.]]
        Y = np.array(Y).T
        # 训练集数据的大小
        date_size = len(X)
        # 定义训练数据batch的大小
        batch_size = 3
        # 定义变量
        w1 = tf.Variable(tf.random_normal([2, 3], stddev=1), name="w1")
        w2 = tf.Variable(tf.random_normal([3, 1], stddev=1), name="w2")
        biases1 = tf.Variable(tf.constant(0.001, shape=[3]), name="b1")# 隐藏层的偏向bias
        biases2 = tf.Variable(tf.constant(0.001,shape=[1]),name="b2")# 输出层的偏向bias
        # 使用placeholder
        x = tf.placeholder(tf.float32, shape=[None, 2], name="x-input")
        y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y-input")
        a = tf.matmul(x, w1) + biases1
        y = tf.matmul(a, w2) + biases2
        cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
        learning_rate = 0.001
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print("w初始：", sess.run(w1))
            # print sess.run(biases1)
            for i in range(100):
                for k in range(0, date_size, batch_size):
                    mini_batch = X[k:k + batch_size]
                    train_y = Y[k:k + batch_size]
                    sess.run(train_step, feed_dict={x: mini_batch, y_: train_y})
                    # print sess.run(w1, feed_dict={x: mini_batch, y_: train_y})
                    # print sess.rund(biases1, feed_dict={x: mini_batch, y_: train_y})
                    if i % 1 == 0:
                        total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
                        print("第%d轮 cross_entropy:%g" % (i, total_cross_entropy))
            print("更新后w1:", sess.run(w1))

if __name__ == "__main__":
    c = TensorflowTest02()
    c.test()