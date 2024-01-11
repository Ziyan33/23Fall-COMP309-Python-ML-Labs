# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:58:02 2023

@author: 14129
"""

# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/14129/Desktop/dataWarehs/"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_ziyan_i = pd.read_csv(fullpath,sep=',')
print(data_ziyan_i)
print(data_ziyan_i.columns.values)
print(data_ziyan_i.shape)
print(data_ziyan_i.describe())
print(data_ziyan_i.dtypes)
print(data_ziyan_i.head(5))
print(data_ziyan_i['Species'].unique())

'''----------------------'''

# Splitting the predictor and target variables
colnames=data_ziyan_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)

# splitting the dataset into train and test variables
import numpy as np
data_ziyan_i['is_train'] = np.random.uniform(0, 1, len(data_ziyan_i)) <= .75
print(data_ziyan_i.head(5))

# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_ziyan_i[data_ziyan_i['is_train']==True], data_ziyan_i[data_ziyan_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

'''----------------------'''

from sklearn.tree import DecisionTreeClassifier

dt_ziyan = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_ziyan.fit(train[predictors], train[target])
'''----------------------'''
preds=dt_ziyan.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])
'''----------------------'''
from sklearn.tree import export_graphviz
with open('C:/Users/14129/Desktop/dataWarehs/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_ziyan, out_file = dotfile, feature_names = predictors)
dotfile.close()
'''----------------------'''
X=data_ziyan_i[predictors]
Y=data_ziyan_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
'''----------------------'''


dt1_ziyan = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=20,
random_state=99)
dt1_ziyan.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_ziyan, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
score

'''----------------------'''
### Test the model using the testing data
testY_predict = dt1_ziyan.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))
'''----------------------'''
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()
