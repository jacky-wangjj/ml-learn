#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------
# Author:   wangjj17
# Name:     TensorflowTest03
# Date:     2019/2/23
# -------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #输出图像的库

class TensorflowTest03:
    def __init__(self):
        pass

    # 加入神经网络层，activation_function是激励函数，初始化为None
    def add_layer(inputs, in_size, out_size, activation_function=None):

        # 定义权重(矩阵),为一个有in_size行，out_size列的矩阵,矩阵元素值取自一个正态分布中的随机数。
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # random_normal

        # 定义偏置(列表)，初始值推荐不为零
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.1

        # Weight*x+biases
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # 如果没有激励函数（线性），直接输出计算结果
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def test(self):
        # 定义两个占位符,元素格式为float32，维度为任意行1列
        xs = tf.placeholder(tf.float32, [None, 1])
        ys = tf.placeholder(tf.float32, [None, 1])

        # 定义隐藏层l1（输出为10个神经元），输入矩阵为xs（为了避免提前重复计算，这里先用占位符填充）
        l1 = TensorflowTest03.add_layer(xs, 1, 10, activation_function=tf.nn.relu)

        # 定义输出层(1个神经元)
        prediction = TensorflowTest03.add_layer(l1, 10, 1, activation_function=None)

        # reduce_mean为取平均，reduce_sum为取和, reduction_indices为1意思是矩阵列化为1
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=1))

        # 每次以0.1的效率对误差进行更正（优化器）
        train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

        x_data = np.linspace(-1, 1, 300)[:,np.newaxis]  # x_data有一个特性，300个例子,元素值的范围是-1到1，[:,np.newaxis]是用来增加维度的，例如这里就把原矩阵从(300，)的一维数组转化成(300,1)的二维数组
        noise = np.random.normal(0, 0.05, x_data.shape)  # noise取值范围是0到0.05，维度和x_data一样
        y_data = np.square(x_data) - 0.5 + noise  # y_data=x_data的平方-0.5+noise;

        # 必须初始化所有变量
        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        # 利用pyplot库图像化training结果
        fig = plt.figure()  # 定义框体
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x_data, y_data)
        plt.ion()
        plt.show()

        for i in range(2000):
            # Training
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if i % 50 == 0:
                print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                prediction_value = sess.run(prediction, feed_dict={xs: x_data})
                lines = ax.plot(x_data, prediction_value, 'r--', lw=5)
                plt.pause(0.2)
        #str = input("执行完毕，输入任意键退出");
        print("执行完毕！")

if __name__ == "__main__":
    c = TensorflowTest03()
    c.test()