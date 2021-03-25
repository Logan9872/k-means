# 引入基础库
import numpy
import sklearn
import scipy
# 引入iris数据库
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import datasets
from sklearn.datasets import load_iris

# 定义数据
iris = load_iris()
# X = iris.data[:]
X = iris.data[:,2:] ##表示我们只取特征空间中的后两个维度
# print(iris)

# 绘制散点图，设置颜色、标记、标签
plt.scatter(X[:, 0], X[:, 1], c="red", marker='.', label='origin')
# 设置XY轴
plt.xlabel('petal length')
plt.ylabel('petal width')
# 创建图例
plt.legend(loc=2)
plt.show()

#构造聚类器
estimator = KMeans(n_clusters=4)
#聚类(fit方法对训练)
estimator.fit(X)
#获取聚类标签
label_pred = estimator.labels_

#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == 3]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='.', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:,1], c= "yellow",marker='.',label='label3')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

