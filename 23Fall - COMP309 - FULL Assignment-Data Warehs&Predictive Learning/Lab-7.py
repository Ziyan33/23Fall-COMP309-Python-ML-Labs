# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:03:56 2023

@author: 14129
"""

# -*- coding: utf-8 -*-

"""

@author: ziyan

"""
'''Q2'''

import pandas as pd

import os

path = "C:/Users/14129/Desktop/dataWarehs"

filename = 'Advertising.csv'

fullpath = os.path.join(path,filename)

data_ziyan_adv = pd.read_csv(fullpath)

data_ziyan_adv.columns.values

data_ziyan_adv.shape

data_ziyan_adv.describe()

data_ziyan_adv.dtypes

data_ziyan_adv.head(5)

'''----------------------'''

# -*- coding: utf-8 -*-
import numpy as np

def corrcoeff(df,var1,var2):

    df['corrn']=(df[var1]-np.mean(df[var1]))*(df[var2]-np.mean(df[var2]))

    df['corrd1']=(df[var1]-np.mean(df[var1]))**2

    df['corrd2']=(df[var2]-np.mean(df[var2]))**2

    corrcoeffn=df.sum()['corrn']

    corrcoeffd1=df.sum()['corrd1']

    corrcoeffd2=df.sum()['corrd2']

    corrcoeffd=np.sqrt(corrcoeffd1*corrcoeffd2)

    corrcoeff=corrcoeffn/corrcoeffd

    return corrcoeff

print(corrcoeff(data_ziyan_adv,'TV','Sales'))

print(corrcoeff(data_ziyan_adv,'Radio','Sales'))

print(corrcoeff(data_ziyan_adv,'Newspaper','Sales'))

'''----------------------'''
import matplotlib.pyplot as plt

plt.plot(data_ziyan_adv['TV'],data_ziyan_adv['Sales'],'ro')

plt.title('TV vs Sales')

plt.plot(data_ziyan_adv['Radio'],data_ziyan_adv['Sales'],'ro')

plt.title('Radio vs Sales')

plt.plot(data_ziyan_adv['Newspaper'],data_ziyan_adv['Sales'],'ro')

plt.title('Newspaper vs Sales')

'''--------------------'''

import statsmodels.formula.api as smf

model1=smf.ols(formula='Sales~TV',data=data_ziyan_adv).fit()

model1.params
'''--------------------'''

print(model1.pvalues)

print(model1.rsquared)

print(model1.summary())
'''--------------------'''

import statsmodels.formula.api as smf

model3=smf.ols(formula='Sales~TV+Radio',data=data_ziyan_adv).fit()

print(model3.params)

print(model3.rsquared)

print(model3.summary())

## Predicte a new value

X_new2 = pd.DataFrame({'TV': [50],'Radio' : [40]})

# predict for a new observation

sales_pred2=model3.predict(X_new2)

print(sales_pred2)
'''--------------------'''
#Better solution than the previous method- test and train split

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

feature_cols = ['TV', 'Radio']

X = data_ziyan_adv[feature_cols]

Y = data_ziyan_adv['Sales']

trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)

lm = LinearRegression()

lm.fit(trainX, trainY)

print (lm.intercept_)

print (lm.coef_)

zip(feature_cols, lm.coef_)

[('TV', 0.045706061219705982), ('Radio', 0.18667738715568111)]

lm.score(trainX, trainY)

lm.predict(testX)


''''--------------------'''
from sklearn.feature_selection import RFE

from sklearn.svm import SVR

feature_cols = ['TV', 'Radio','Newspaper']

X = data_ziyan_adv[feature_cols]

Y = data_ziyan_adv['Sales']

estimator = SVR(kernel="linear")

selector = RFE(estimator,step=1)
#selector = RFE(estimator,2,step=1)

selector = selector.fit(X, Y)

print(selector.support_)

print(selector.ranking_)

'''------------------------------------'''
'''------------------------------------'''
# -*- coding: utf-8 -*-
"""
@author: ziyan
"""
import pandas as pd
import os
path = "C:/Users/14129/Desktop/dataWarehs"
filename = 'Bank.csv'
fullpath = os.path.join(path,filename)
data_ziyan_b = pd.read_csv(fullpath,sep=';')
print(data_ziyan_b.columns.values)
print(data_ziyan_b.shape)
print(data_ziyan_b.describe())
print(data_ziyan_b.dtypes)
print(data_ziyan_b.head(5))

'''----------------------------------'''

# -*- coding: utf-8 -*-
"""
@author: ziyan
"""

import pandas as pd
import os
path = "C:/Users/14129/Desktop/dataWarehs"
filename = 'Bank.csv'
fullpath = os.path.join(path,filename)
data_ziyan_b = pd.read_csv(fullpath,sep=';')

print(data_ziyan_b.columns.values)
print(data_ziyan_b.shape)
print(data_ziyan_b.describe())
print(data_ziyan_b.dtypes)
print(data_ziyan_b.head(5))
print(data_ziyan_b['education'].unique())

import numpy as np
data_ziyan_b['education']=np.where(data_ziyan_b['education'] =='basic.9y', 'Basic',
data_ziyan_b['education'])
data_ziyan_b['education']=np.where(data_ziyan_b['education'] =='basic.6y', 'Basic',
data_ziyan_b['education'])
data_ziyan_b['education']=np.where(data_ziyan_b['education'] =='basic.4y', 'Basic',
data_ziyan_b['education'])
data_ziyan_b['education']=np.where(data_ziyan_b['education'] =='university.degree', 'UniversityDegree', data_ziyan_b['education'])
data_ziyan_b['education']=np.where(data_ziyan_b['education'] =='professional.course', 'ProfessionalCourse', data_ziyan_b['education'])
data_ziyan_b['education']=np.where(data_ziyan_b['education'] =='high.school', 'High School',
data_ziyan_b['education'])
data_ziyan_b['education']=np.where(data_ziyan_b['education'] =='illiterate', 'Illiterate',
data_ziyan_b['education'])
data_ziyan_b['education']=np.where(data_ziyan_b['education'] =='unknown', 'Unknown',
data_ziyan_b['education'])
#Check the values of who purchased the deposit account
print(data_ziyan_b['y'].value_counts())


#Check the average of all the numeric columns
pd.set_option('display.max_columns',100)
print(data_ziyan_b.groupby('y').mean(numeric_only=True))


#Check the mean of all numeric columns grouped by education
print(data_ziyan_b.groupby('education').mean(numeric_only=True))


#Plot a histogram showing purchase by education category
import matplotlib.pyplot as plt
pd.crosstab(data_ziyan_b.education,data_ziyan_b.y)
pd.crosstab(data_ziyan_b.education,data_ziyan_b.y).plot(kind='bar')
plt.title('Purchase Frequency for Education Level')
plt.xlabel('Education')
plt.ylabel('Frequency of Purchase')
#draw a stacked bar chart of the marital status and the purchase of term deposit to see whether this can be a good predictor of the outcome
table=pd.crosstab(data_ziyan_b.marital,data_ziyan_b.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')

#plot the bar chart for the Frequency of Purchase against each day of the week to see whether this can be a good predictor of the outcome
pd.crosstab(data_ziyan_b.day_of_week,data_ziyan_b.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
#Repeat for the month
pd.crosstab(data_ziyan_b.month,data_ziyan_b.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
#Plot a histogram of the age distribution
data_ziyan_b.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

#Deal with the categorical variables, use a for loop
#1- Create the dummy variables
###############################################################################
#######
import pandas as pd
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(data_ziyan_b[var], prefix=var)
    data_ziyan_b1=data_ziyan_b.join(cat_list)
    data_ziyan_b=data_ziyan_b1
    data_ziyan_b.head(2)

# 2- Remove the original columns
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_ziyan_b_vars=data_ziyan_b.columns.values.tolist()
to_keep=[i for i in data_ziyan_b_vars if i not in cat_vars]
data_ziyan_b_final=data_ziyan_b[to_keep]
data_ziyan_b_final.columns.values
# 3- Prepare the data for the model build as X (inputs, predictor) and Y(output, predicted)
data_ziyan_b_final_vars=data_ziyan_b_final.columns.values.tolist()
Y=['y']
X=[i for i in data_ziyan_b_final_vars if i not in Y ]
type(Y)
type(X)

'''--------------------------'''

#1- We have many features so let us carryout feature selection from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 12)
rfe = rfe.fit(data_ziyan_b_final[X],data_ziyan_b_final[Y] )
print(rfe.support_)
print(rfe.ranking_)

#2- Update X and Y with selected features
cols=['previous', 'euribor3m', 'job_entrepreneur', 'job_self-employed', 'poutcome_success',
'poutcome_failure', 'month_oct', 'month_may',
'month_mar', 'month_jun', 'month_jul', 'month_dec']
X=data_ziyan_b_final[cols]
Y=data_ziyan_b_final['y']
type(Y)
type(X)

'''-----------------------'''

#1- split the data into 70%training and 30% for testing, note added the solver to avoidwarnings
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# 2-Let us build the model and validate the parameters
from sklearn import linear_model
from sklearn import metrics
clf1 = linear_model.LogisticRegression(solver='lbfgs')
clf1.fit(X_train, Y_train)
#3- Run the test data against the new model
probs = clf1.predict_proba(X_test)
print(probs)
predicted = clf1.predict(X_test)
print (predicted)
#4-Check model accuracy
print (metrics.accuracy_score(Y_test, predicted))

'''----------------------'''

from sklearn.model_selection import cross_val_score
scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), X, Y,scoring='accuracy', cv=10)
print (scores)
print (scores.mean())

'''------------------'''
prob=probs[:,1]
prob_df=pd.DataFrame(prob)
prob_df['predict']=np.where(prob_df[0]>=0.05,1,0)
import numpy as np
Y_A =Y_test.map({'no':0,'yes':1}).values.astype(int)
Y_P = np.array(prob_df['predict'])
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_A, Y_P)
print (confusion_matrix)