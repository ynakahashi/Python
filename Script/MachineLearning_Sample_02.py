# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 15:19:55 2015

@author: 140282
"""
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier ,GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets
from sklearn.cross_validation import cross_val_score


# prepare learning dataset
iris     = datasets.load_iris()
features = iris.data
labels   = iris.target

# reduction of feature-dimentions
lsa = TruncatedSVD(2)
reduced_features = lsa.fit_transform(features)

#どのモデルがいいのかよくわからないから目があったやつとりあえずデフォルト設定で全員皆殺し 
clf_names = ["LinearSVC","AdaBoostClassifier","ExtraTreesClassifier" ,"GradientBoostingClassifier","RandomForestClassifier"]
for clf_name in clf_names:
  clf    = eval("%s()" % clf_name)
  scores = cross_val_score(clf,reduced_features, labels,cv=5)
  score  = sum(scores) / len(scores)  #モデルの正解率を計測
  print "%sのスコア:%s" % (clf_name,score)

#LinearSVCのスコア:0.973333333333
#AdaBoostClassifierのスコア:0.973333333333
#ExtraTreesClassifierのスコア:0.973333333333
#GradientBoostingClassifierのスコア:0.966666666667
#RandomForestClassifierのスコア:0.933333333333
