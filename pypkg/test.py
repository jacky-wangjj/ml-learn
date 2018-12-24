#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : test
# @Time     : 2018/12/24
# ------------------------
import calendar
import time

class test:
    def __init__(self):
        pass

    def main(self):
        print "hello world";
        # print raw_input("put enter exit\n"); #等待用户输入 enter结束
        counter = 100  # 整型变量
        miles = 1000.0  # 浮点型
        name = "John"  # 字符串
        print counter, miles, name
        s = 'abcdefgh'
        print s[1:5]  # 从前向后索引
        print s[-6:-1]  # 从后向前索引
        tuple = ('runoob', 786, 2.33, 'john', 79.2)  # tuple,不允许更新
        print tuple[1:3]
        flag = False
        if name == 'John':
            flag = True
            print 'welcome boss'
        elif name == 'john':
            print 'john'
        else:
            print name
        print flag

    def listTest(self):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];  # list,允许更新
        print days[0:5]
        for day in days:
            print day,
        print "end"
        for index in range(len(days)):
            print days[index],
        print "end"
        days.append('Saturday')  # 列表更新元素
        days.append('Sunday')
        print days
        del days[6]  # 列表删除元素
        print days
        print days * 2  # 重复列表
        print days[1:]  # 列表截取
        print len(days)
        print max(days)
        print days.pop()  # 移除最后一个元素
        print days
        days.reverse()  # 列表反转
        print days
        days.sort()  # 列表排序
        print days

    def dictTest(self):
        dict = {'runoob': '菜鸟教程', 'google': 'Google 搜索'}
        print dict.keys()  # 输出所有键
        print dict.values()  # 输出所有值
        print dict['name']
        dict['school'] = 'qinghua'
        print dict
        print str(dict)
        print type(dict)
        dict.clear()
        print 'len(dict):%d' % len(dict)
        print "Value : %s" % dict.setdefault('runoob', None)
        print "Value : %s" % dict.setdefault('Taobao', '淘宝')
        print str(dict)

    def getTime(self):
        ticks = time.time()
        print 'timestramp:%ld' % ticks
        localtime = time.localtime(time.time())
        print "本地时间：", localtime
        print "本地时间：", time.asctime(localtime)
        print time.strftime("%Y-%m-%d %H:%M:%S %W %Z", time.localtime())

    def getCalendar(self):
        cal = calendar.month(2018, 12)
        print cal

    def printinfo(self, name, age=35, *vars):
        print 'name:', name, ' age:', age
        for var in vars:
            print 'var:', var,
        sum = lambda arg1, arg2: arg1 + arg2;
        print sum(10, 20)
        return;

    def commonFunc(self):
        seq = ['one', 'two', 'three']
        for i, element in enumerate(seq):
            print i, element


if __name__ == "__main__":
    c = test()
    t0 = time.clock()
    c.main()
    c.getTime()
    c.getCalendar()
    time.sleep(2)
    print time.clock() - t0
    c.printinfo(name='jacky')
    c.printinfo('liming', 20, 'var1', 'var2')
    print dir(time)
    c.commonFunc()