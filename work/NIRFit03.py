#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------
# @Author   : wangjj17
# @File     : NIRFit03
# @Time     : 2019/2/28
# ------------------------
import numpy as np
from sklearn import linear_model, svm, neighbors, ensemble
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import SavitzkyGolay

class NIRFit03:
    def __init__(self):
        pass

    fileName = "dataset2.CSV"
    # 第1列是浓度，2-1558列是不同波长的吸光度
    # 第1行是光谱数据的不同波长，2-11行是不同浓度的多组光谱数据
    def loadDataSet(fileName):
        numFeat = len(open(fileName).readline().split(',')) - 1
        print("数据集维度为：", numFeat)
        dataMat = [];
        labelMat = []
        fr = open(fileName)
        lineNum = 0
        for line in fr.readlines():
            lineArr = []
            lineNum += 1
            if (lineNum != 1):
                curLine = line.strip().split(',')
                for i in range(1, numFeat + 1):
                    lineArr.append(float(curLine[i]))
                dataMat.append(lineArr)
                labelMat.append(float(curLine[0]))
        # 对数据集中的数据进行排序
        sort_index = np.argsort(labelMat)
        # print(sort_index)
        sort_labelMat = np.array(labelMat)[sort_index]
        sort_dataMat = np.array(dataMat)[sort_index,:]
        # print(sort_labelMat)
        return sort_dataMat, sort_labelMat

    # 多元散射校正
    def multiplicative_scatter_correction(xArr):
        xArr = np.array(xArr)
        m,n = xArr.shape
        print(m,n)
        xMean = np.mean(xArr, axis=0)
        print("xArr各波长吸光度均值：\n", xMean)
        # xArr_msc = xArr.copy()
        for i in range(1,m):
            k,b = np.polyfit(xMean, xArr[i,:], 1)
            # print("回归系数为：\n", k)
            # print("截距为：\n", b)
            xArr[i,:] = (xArr[i,:]-b)/k
        return xArr

    # 相关系数，K为选择的维度
    def correlation_coefficient(xArr, yArr, K):
        xArr = np.array(xArr)
        m,n = xArr.shape
        coef = [0 for x in range(n)]
        for i in range(n):
            coef[i] = np.corrcoef(xArr[:,i], yArr)[0,1]
        print("相关系数：\n", coef)
        print("最大相关系数：\n", np.max(coef))
        print("最小相关系数：\n", np.min(coef))
        # 选取最相关的K个维度
        coef_index = np.argsort(np.abs(np.array(coef)))[-K:]
        xArr = xArr[:,coef_index]
        print("相关系数筛选后的维度：\n", xArr.shape)
        return xArr

    # SavitzkyGolay平滑滤波
    # window_size为窗口大小，rank拟合多项式阶次
    def savitzk_golay(xArr, window_size, rank):
        xArr = np.array(xArr)
        m,n = xArr.shape
        xArr_sg = []
        for i in range(m):
            xArr_sg.append(SavitzkyGolay.savgol(xArr[i,:], window_size, rank))
        return np.array(xArr_sg)

    def get_data(fileName):
        xArr, yArr = NIRFit03.loadDataSet(fileName)
        # 画出数据集光谱各波长对应的吸光度
        NIRFit03.drawNIR(xArr, yArr)
        # 画出数据集光谱中每个维度波长吸光度与浓度的关系，n为第几维
        # NIRFit03.drawEachNIR(np.array(xArr), np.array(yArr), np.argmax(xArr[0,:]))
        # 手动剔除异常数据，删除指定行，画出剔除异常数据后数据集光谱各波长对应的吸光度
        # xArr = np.delete(np.array(xArr), 1, 0)
        # yArr = np.delete(np.array(yArr), 1, 0)
        # NIRFit03.drawNIR(xArr, yArr)
        # 手动选取维度，画出选取维度后数据集光谱各波长对应的吸光度
        xArr = np.array(xArr)[:,20:len(xArr[0])]
        print("选取的维度为：", len(xArr[0]))
        NIRFit03.drawNIR(xArr, yArr)
        # 多元散射校正
        xArr = NIRFit03.multiplicative_scatter_correction(xArr)
        NIRFit03.drawNIR(xArr, yArr)
        # savitzky golay平滑滤波，窗口越大越平滑，但易失真，阶数越低越平滑
        xArr = NIRFit03.savitzk_golay(xArr, 19, 3)
        NIRFit03.drawNIR(xArr, yArr)
        # 画出数据集光谱中每个维度波长吸光度与浓度的关系，n为第几维
        NIRFit03.drawEachNIR(np.array(xArr), np.array(yArr), np.argmax(xArr[0, :]))
        # 相关系数选取特征波长，选取相关系数绝对值最大的n个维度
        xArr = NIRFit03.correlation_coefficient(xArr, yArr, 300)
        NIRFit03.drawNIR(xArr, yArr)
        # 特征选择
        # xArr = NIRFit03.featureSelection(np.array(xArr), np.array(yArr).ravel())
        # NIRFit03.drawNIR(xArr, yArr)
        # PCA降维，由入参指定降到的维数，画出降维后数据集光谱各波长对应的吸光度
        # xArr = NIRFit03.pca_op(xArr, 820)
        # xArr = NIRFit03.pca_op(xArr, 0.99)
        # NIRFit03.drawNIR(xArr, yArr)
        # array转mat
        xMat = np.mat(xArr);
        yMat = np.mat(yArr).T
        # 拆分集合为训练集合测试集
        # x_train, x_test, y_train, y_test = train_test_split(xMat, yMat, test_size=0.2)
        # 固定训练集和测试集
        index_train = [0,1,2,4,5,6,7,8,9,11,13,15,16]
        index_test = [3,10,12,14]
        x_train = xMat[index_train, :]
        y_train = yMat[index_train, :]
        x_test = xMat[index_test, :]
        y_test = yMat[index_test, :]
        # 标准化
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        print('X train set:\n', x_train)
        print('X test set:\n', x_test)
        print('Y train set:\n', y_train)
        print('Y test set:\n', y_test)
        return x_train, y_train, x_test, y_test

    # 画出数据集光谱中每个维度波长吸光度与浓度的关系
    def drawEachNIR(xArr, yArr, n):
        plt.figure()
        plt.scatter(xArr[:,n], yArr, color='r', label='NIR-C')
        plt.title('each NIR data')
        plt.xlabel('absorbance')
        plt.ylabel('concentration')
        plt.legend()
        plt.show()

    # 画出数据集光谱各波长对应的吸光度
    def drawNIR(xArr, yArr):
        plt.figure()
        for i in range(len(yArr)):
            plt.plot(np.arange(len(xArr[i])), xArr[i], '-o', label=yArr[i])
        plt.title('NIR data')
        plt.xlabel('wavelength')
        plt.ylabel('absorbance')
        plt.legend()
        plt.show()

    # 回归系数的分布直方图，可查看是否能降维
    def drawCoef(coef):
        f = plt.figure(figsize=(7, 5))
        ax = f.add_subplot(111)
        ax.hist(coef.T, bins=50, color='b')
        ax.set_title("Histogram of the regr.coef_")
        plt.show()

    # 画出测试集预测Y与真实Y的差距
    def drawPred(y_pred, y_test, score):
        # the plot
        plt.figure()
        plt.plot(np.arange(len(y_pred)), y_test, 'go-', label='true value')
        plt.plot(np.arange(len(y_pred)), y_pred, 'ro-', label='predict value')
        plt.title('score:%f' % score)
        plt.xlabel('test index')
        plt.ylabel('y value')
        plt.legend()
        plt.show()

    # 特征选择
    def featureSelection(X, Y):
        # 方差选择法
        # X_sel = VarianceThreshold(threshold=0.8).fit_transform(X)
        # 卡方检验
        X_sel = SelectKBest(f_regression, k=900).fit_transform(X, Y)
        print('X shape:\n', X_sel.shape)
        return X_sel

    # PCA降维
    def pca_op(xArr, n):
        pca = PCA(n_components=n, whiten=False, svd_solver='auto')
        pca.fit(xArr)
        red_X = pca.transform(xArr)
        print("降维后特征数：", pca.n_components_)
        print('保留方差百分比：', pca.explained_variance_ratio_.sum())
        return red_X

    # 测试各种回归算法，输出相关信息
    def try_different_method(clf, x_train, y_train, x_test, y_test):
        if isinstance(clf, svm.SVR) or isinstance(clf, ensemble.RandomForestRegressor)\
                or isinstance(clf, ensemble.AdaBoostRegressor) or isinstance(clf, ensemble.GradientBoostingRegressor):
            y_train = np.array(y_train).ravel()
            y_test = np.array(y_test).ravel()
        model = clf.fit(x_train, y_train)
        print('model: \n', model)
        score = clf.score(x_test, y_test)
        y_pred = clf.predict(x_test)
        print('predict Y: \n', y_pred)
        if isinstance(clf, linear_model.LinearRegression):
            # The coefficients
            coef = clf.coef_
            print('Coefficients: \n', coef)
            NIRFit03.drawCoef(coef)
            # The Intercepts
            intercept = clf.intercept_
            print('Intercepts: \n', intercept)
        # score
        print("score: %.4f" % clf.score(x_test, y_test))
        # The mean squared error, 均方差
        print("Mean squared error: %.4f" % mean_squared_error(y_test, y_pred))
        # The mean absolute error, 平均绝对误差
        print('Mean absolute error: %.4f' % mean_absolute_error(y_test, y_pred))
        # The median absolute error, 中值绝对误差
        print('Median absoulte error: %.4f' % median_absolute_error(y_test, y_pred))
        # Explained variance score: 1 is perfect prediction; r2->0模型越差，r2->1模型越好
        # R2决定系数，表征回归方程在多大程度上解释了因变量的变化，或者说方程对观测值的拟合程度如何。
        print('Variance score: %.4f' % r2_score(y_test, y_pred))
        # The cross validation
        X = np.vstack((x_train, x_test))
        if isinstance(clf, svm.SVR) or isinstance(clf, ensemble.RandomForestRegressor)\
                or isinstance(clf, ensemble.AdaBoostRegressor) or isinstance(clf, ensemble.GradientBoostingRegressor):
            Y = np.hstack((y_train, y_test))
        else:
            Y = np.vstack((y_train, y_test))
        scores = cross_val_score(clf, X, Y, cv=int(len(Y)*0.8))
        print("Cross val score: \n", scores)
        print("Mean score: %.4f" % scores.mean())
        # The cross predict
        print("Cross val predicte:\n", cross_val_predict(clf, X, Y, cv=int(len(Y)*0.9)))
        # the plot
        NIRFit03.drawPred(y_pred, y_test, score)
        NIRFit03.drawPred(clf.predict(X), Y, score)

    # 偏最小二乘
    def PLSRegressionTest(self):
        x_train, y_train, x_test, y_test = NIRFit03.get_data(NIRFit03.fileName)
        n_components = 0
        scores = [0 for x in range(x_train.shape[1])]
        while n_components < x_train.shape[1]:
            n_components+=1
            plsg = PLSRegression(n_components=n_components)
            plsg.fit(x_train, y_train)
            scores[n_components-1] = plsg.score(x_test, y_test)
        xx = np.linspace(1, len(scores), len(scores))
        plt.figure()
        plt.plot(xx, scores, 'o-')
        plt.show()
        print('scores:\n', scores)
        # 选取使得score最大的n_components进行最小二乘建模
        plsg2 = PLSRegression(n_components=np.argmax(scores)+1)
        NIRFit03.try_different_method(plsg2, x_train, y_train, x_test, y_test)

    # 回归树
    def DecisionTreeRegressTest(self):
        x_train, y_train, x_test, y_test = NIRFit03.get_data(NIRFit03.fileName)
        clf = DecisionTreeRegressor()
        NIRFit03.try_different_method(clf, x_train, y_train, x_test, y_test)

    # 线性回归
    def LinearRegressionTest(self):
        x_train, y_train, x_test, y_test = NIRFit03.get_data(NIRFit03.fileName)
        linear_reg = linear_model.LinearRegression()
        NIRFit03.try_different_method(linear_reg, x_train, y_train, x_test, y_test)

    # SVM支持向量机
    def SVMTest(self):
        x_train, y_train, x_test, y_test = NIRFit03.get_data(NIRFit03.fileName)
        svr = svm.SVR()
        NIRFit03.try_different_method(svr, x_train, y_train, x_test, y_test)

    # K邻近knn
    def KNeighborsRegressorTest(self):
        x_train, y_train, x_test, y_test = NIRFit03.get_data(NIRFit03.fileName)
        knn = neighbors.KNeighborsRegressor()
        NIRFit03.try_different_method(knn, x_train, y_train, x_test, y_test)

    # 随机森林
    def RandomForestRegressorTest(self):
        x_train, y_train, x_test, y_test = NIRFit03.get_data(NIRFit03.fileName)
        rf = ensemble.RandomForestRegressor(n_estimators=20)#使用20个决策树
        NIRFit03.try_different_method(rf, x_train, y_train, x_test, y_test)

    # Adaboost自适应增强
    def AdaboostTest(self):
        x_train, y_train, x_test, y_test = NIRFit03.get_data(NIRFit03.fileName)
        ada = ensemble.AdaBoostRegressor(n_estimators=50)
        NIRFit03.try_different_method(ada, x_train, y_train, x_test, y_test)

    # GBRT渐进梯度回归树
    def GradientBoostingRegressorTest(self):
        x_train, y_train, x_test, y_test = NIRFit03.get_data(NIRFit03.fileName)
        gbrt = ensemble.GradientBoostingRegressor(n_estimators=100)
        NIRFit03.try_different_method(gbrt, x_train, y_train, x_test, y_test)

    # 多项式回归
    def PolynominalRegressorTest(self):
        x_train, y_train, x_test, y_test = NIRFit03.get_data(NIRFit03.fileName)
        featurizer = PolynomialFeatures(degree=2) #degree定义是最高次项
        x_train_featurizer = featurizer.fit_transform(x_train) #fit_transform：正则化
        x_test_featurizer = featurizer.fit_transform(x_test) #fit_transform：正则化
        lr = linear_model.LinearRegression()
        NIRFit03.try_different_method(lr, x_train_featurizer, y_train, x_test_featurizer, y_test)

if __name__ == "__main__":
    c = NIRFit03()
    # c.DecisionTreeRegressTest()
    # c.LinearRegressionTest()
    # c.SVMTest()
    # c.KNeighborsRegressorTest()
    # c.RandomForestRegressorTest()
    # c.AdaboostTest()
    # c.GradientBoostingRegressorTest()
    # c.PolynominalRegressorTest()
    c.PLSRegressionTest()