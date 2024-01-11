# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:38:38 2023

@author: 14129
"""
from sklearn.datasets import load_iris
from sklearn import tree
iris=load_iris()
print(iris)


X, y = load_iris(return_X_y=True)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf)


####################################################
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz

iris = load_iris()
x = iris.data
y=iris.target

tree_clf = DecisionTreeClassifier()
model = tree_clf.fit(x,y)

dot_data = export_graphviz(tree_clf,out_file=None,feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True,special_characters=True)

graph = graphviz.Source(dot_data)
#graph.render("iris1")

prob = tree_clf.predict_proba([[7,3.3,4.5,1.5]])

print(y)
print(iris)
print(prob)
