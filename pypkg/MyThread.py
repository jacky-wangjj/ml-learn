#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : MyThread
# @Time     : 2018/12/24
# ------------------------
import threading
import time

exitFlag = 0
class MyThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        pass

    def run(self):
        print("starting "+self.name)
        threadLock.acquire() #获取锁（同步）
        print_time(self.name, self.counter, 5)
        threadLock.release() #释放锁
        print("exit "+self.name)

threadLock = threading.Lock()
threads = []
def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            (threading.Thread).exit()
        time.sleep(delay)
        print("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1

if __name__ == "__main__":
    thread1 = MyThread(1, "thread-1", 1)
    thread2 = MyThread(2, "thread-2", 2)
    thread1.start()
    thread2.start()
    threads.append(thread1)
    threads.append(thread2)
    for t in threads:
        t.join()
    print("exit main thread")