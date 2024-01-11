# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:14:21 2023

@author: 14129
"""

import pandas as pd
import os
path = "C:/Users/14129/Desktop/dataWarehs"

filename = 'wine.csv'
fullpath = os.path.join(path,filename)
data_ziyan_wine = pd.read_csv(fullpath,sep=';')

print (data_ziyan_wine)

## Setting display option in pandas to show a maximum of 15 columns

pd.set_option('display.max_columns',15)
print(data_ziyan_wine.head())
print(data_ziyan_wine.columns.values)
print(data_ziyan_wine.shape)
print(data_ziyan_wine.describe())
print(data_ziyan_wine.dtypes)
print(data_ziyan_wine.head(5))



# Grouping the DataFrame by the 'quality' column and calculating the mean of each group
print(data_ziyan_wine['quality'].value_counts())
# number_quality=data_ziyan_wine['quality'].value_counts()
# print("number of items ",number_quality)


print(data_ziyan_wine['quality'].unique())
unique_items=data_ziyan_wine['quality'].unique()
print("Unique items",unique_items)


pd.set_option('display.max_columns',15)
print(data_ziyan_wine.groupby('quality').mean())


'''
***************************************************************

***************************************************************
-'''
import matplotlib.pyplot as plt

# Creating a histogram of the 'quality' column
plt.hist(data_ziyan_wine['quality'])

'''-------------------------------'''
#Use seaborn library to generate different plots:
import seaborn as sns

# Creating a distribution plot for the 'quality' column
sns.distplot(data_ziyan_wine['quality'])


# plot only the density function
sns.distplot(data_ziyan_wine['quality'], rug=True, hist=False, color = 'g')


# Change the direction of the plot
sns.distplot(data_ziyan_wine['quality'], rug=True, hist=False, vertical = True)


# Check all correlations. Here it take longer time to execute
sns.pairplot(data_ziyan_wine)


# Subset three column
x=data_ziyan_wine[['fixed acidity','chlorides','pH']]
y=data_ziyan_wine[['chlorides','pH']]

# check the correlations
sns.pairplot(x)




# Generate heatmaps
sns.heatmap(data_ziyan_wine[['fixed acidity']])
sns.heatmap(x)

# Generating heatmap for the correlation matrix
sns.heatmap(x.corr())
# Generating annotated heatmap for the correlation matrix
sns.heatmap(x.corr(),annot=True)


import matplotlib.pyplot as plt
# Creating an annotated heatmap with specific styling
plt.figure(figsize=(10,9))
sns.heatmap(x.corr(),annot=True, cmap='coolwarm',linewidth=0.5)


# Creating line plots for the selected variables
##line two variables
plt.figure(figsize=(20,9))
sns.lineplot(data=y)
sns.lineplot(data=y,x='chlorides',y='pH')


## line three variables
sns.lineplot(data=x)

data_ziyan_wine_norm=(data_ziyan_wine - data_ziyan_wine.min())/(data_ziyan_wine)
data_ziyan_wine_norm.head()


# check some plots after normalizing the data
x1=data_ziyan_wine_norm[['fixed acidity','chlorides','pH']]
y1=data_ziyan_wine_norm[['chlorides','pH']]
sns.lineplot(data=y1)
sns.lineplot(data=x1)
sns.lineplot(data=y,x='chlorides',y='pH')


'''-------------------------------'''
#Normalize the data in order to apply clustering
data_ziyan_wine_norm = (data_ziyan_wine - data_ziyan_wine.min()) / (data_ziyan_wine.max() -
data_ziyan_wine.min())
data_ziyan_wine_norm.head()

'''-------------------------------'''

# check some plots after normalizing the data
x1=data_ziyan_wine_norm[['fixed acidity','chlorides','pH']]
y1=data_ziyan_wine_norm[['chlorides','pH']]
sns.lineplot(data=y1)
sns.lineplot(data=x1)
sns.lineplot(data=y,x='chlorides',y='pH')

'''-------------------------------'''

from sklearn.cluster import KMeans
#from sklearn import datasets
model=KMeans(n_clusters=6)
model.fit(data_ziyan_wine_norm)



'''-------------------------------'''
model.labels_
# Append the clusters to each record on the dataframe, i.e. add a new column for clusters
md=pd.Series(model.labels_)
data_ziyan_wine_norm['clust']=md
data_ziyan_wine_norm.head(10)
#find the final cluster's centroids for each cluster
model.cluster_centers_
#Calculate the J-scores The J-score can be thought of as the sum of the squared distance
#between points and cluster centroid for each point and cluster.
#For an efficient cluster, the J-score should be as low as possible.
model.inertia_
#let us plot a histogram for the clusters
import matplotlib.pyplot as plt
plt.hist(data_ziyan_wine_norm['clust'])
plt.title('Histogram of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
# plot a scatter
plt.scatter(data_ziyan_wine_norm['clust'],data_ziyan_wine_norm['pH'])
plt.scatter(data_ziyan_wine_norm['clust'],data_ziyan_wine_norm['chlorides'])