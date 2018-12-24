#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : Employee
# @Time     : 2018/12/24
# ------------------------

class Employee:
    def __init__(self):
        pass

    empCount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1
        pass

    def displayCount(self):
        print "total employee %d" % Employee.empCount

    def displayEmployee(self):
        print "name:", self.name, "salary:", self.salary

    def __del__(self):
        print self.__class__.__name__, "destroy"

if __name__ == "__main__":
    c1 = Employee("jacky", 20000)
    c1.displayCount()
    c1.displayEmployee()
    c2 = Employee("John", 10000)
    print hasattr(c2, "name"), getattr(c2, "name")
    setattr(c2, "salary", 15000)
    c2.displayEmployee()
    print "Employee.__doc__", Employee.__doc__
    print "Employee.__name__", Employee.__name__
    print "Employee.__module__", Employee.__module__
    print "Employee.__bases__", Employee.__bases__
    print "Employee.__dict__", Employee.__dict__
    del c1
    del c2