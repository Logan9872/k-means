# 月亮数据集的K-means分类
# 月亮数据集的K-means分类
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans·
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.35, random_state=42)
model = KMeans(n_clusters=2)  # 构造聚类器
model.fit(X)  # 聚类
label_pred = model.labels_  # 获取聚类标签
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
# x2 = X[label_pred == 2]
# x3 = X[label_pred == 3]

# plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
# plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x1[:, 0], x1[:, 1], c="blue", marker='s', label='label2')
plt.scatter(x1[:, 0], x1[:, 1], c="yellow", marker='o', label='label3')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=4)
plt.show()
