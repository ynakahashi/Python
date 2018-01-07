# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 13:49:17 2015

@author: 140282
"""
import sklearn as skl
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
import pandas

## SVR(Support Vector Regression)
# 学習データ
np.random.seed(100)
x = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(x).ravel()
# 最後の 5 は刻み幅。
# 0, 5, 10, 15・・・番目のデータに右辺を加える。
y[::5] += 3 * (0.5 - numpy.random.rand(8))
plt.plot(x, y, "o")



from sklearn.svm import SVR
# 学習器の作成
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

# fitで学習，predictで予測
y_rbf = svr_rbf.fit(x, y).predict(x)
y_lin = svr_lin.fit(x, y).predict(x)
y_poly = svr_poly.fit(x, y).predict(x)

plt.plot(x, y, "x", color="r")
plt.plot(x, y_rbf, "o")
plt.plot(x, y_lin, "o")
plt.plot(x, y_poly, "o")



## SVM(Support Vector Machine, ただしsklearnではSupport Vector Classifier)
# 学習データ作成
np.random.seed(100)
X = np.random.randn(300, 2)
# xor は排他的論理和。インプットが異なる場合に True を返す。同値なら False を返す
Y = np.logical_xor(X[:,0]>0, X[:,1]>0)
# どのようなデータになっているか見てみる
Y_df = pandas.DataFrame(Y)
Y_df.head()

from sklearn.svm import SVC
# 分類器の作成
clf = SVC(kernel='rbf', C=1e3, gamma=0.1)
# 学習
clf.fit(X, Y)

# 決定関数までの距離を計算
xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# グラフ化
plt.imshow(Z, interpolation='nearest', extent=[xx.min(),
    xx.max(),
    yy.min(),
    yy.max()],
    aspect='auto',
    origin='lower',
#    cmap=cm.PuOr_r
    )

ctr = plt.contour(xx, yy, Z, levels=[0], linetypes='--')
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.axis([xx.min(), xx.max(), yy.min(), yy.max()])
plt.show()