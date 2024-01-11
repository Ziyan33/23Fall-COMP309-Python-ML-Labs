# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:55:09 2023

@author: 14129
"""
'''----------------------------------'''
'''Q1----------------------------------'''

import pandas as pd
import os
path = "C:/Users/14129/Desktop/dataWarehs"
filename = 'titanic.csv'
fullpath = os.path.join(path,filename)
## read
data_ziyan = pd.read_csv(fullpath)
print (data_ziyan.head(10))# Printing the first 10 rows of the DataFrame

'''----------------------------------'''
'''Q2----------------------------------'''

import pandas as pd
data1_ziyan = pd.read_csv('C:/Users/14129/Desktop/dataWarehs/Customer Churn Model.txt')
data1_ziyan.columns.values# Accessing column names
print(data1_ziyan.columns.values)  # Printing column names
data1_ziyan.dtypes # Getting the data types of each column

# Looping through each column name and printing it
for col in data1_ziyan.columns:
    print(col)
    
'''------------------------------------ '''
'''Q3----------------------------------'''

'''Read line by line below the code, change my firstname to your firstname::'''

data=open('C:/Users/14129/Desktop/dataWarehs/Customer Churn Model.txt','r')

cols=data.readline().strip().split(',') # Reads the first line to get column names
no_cols=len(cols) # Counts the number of columns
print(no_cols)


#### Finding the number of rows
counter=0
main_dict={}
for col in cols:
    main_dict[col]=[] # Initializes a list for each column
    print(main_dict)
    
for line in data:
    values = line.strip().split(',')

for i in range(len(cols)):
    main_dict[cols[i]].append(values[i])
    counter += 1
    
    
print ("The dataset has %d rows and %d columns" % (counter,no_cols))

import pandas as pd
df_ziyan=pd.DataFrame(main_dict)
print (df_ziyan.head(15))

'''----------------------------------------'''
'''Q4----------------------------------'''

import pandas as pd
url='https://gist.githubusercontent.com/kevin336/acbb2271e66c10a5b73aacf82ca82784/raw/e38afe62e088394d61ed30884dd50a6826eee0a8/employees.csv'
medal_data_ziyan=pd.read_csv(url)
print (medal_data_ziyan.head(5))

'''----------------------------------------'''
'''------------------Second Part----------------------'''
'''----------------------------------------'''


'''Q1----------------------------------'''

import pandas as pd
data_ziyan=pd.read_csv('C:/Users/14129/Desktop/dataWarehs/titanic.csv')
######### get first five records
data_ziyan.head(5)

######### get the shape of data
data_ziyan.shape

######## get the column values/ column names
data_ziyan.columns.values
# or

print(data_ziyan.columns.values)
# or

for col in data_ziyan.columns:
    print(col)

###### create summaries of data
data_ziyan.describe()

##### get the data types of columns
data_ziyan.dtypes


'''----------------------------------------'''
'''Q2----------------------------------'''

####Imputation
# Fill the missing values with zeros
import pandas as pd

data_ziyan=pd.read_csv('C:/Users/14129/Desktop/dataWarehs/titanic.csv')
data_ziyan.fillna(0,inplace=True)
data_ziyan.head()


# Fill the missing values with "missing"
import pandas as pd
data_ziyan=pd.read_csv('C:/Users/14129/Desktop/dataWarehs/titanic.csv')
data_ziyan.fillna("missing",inplace=True)
data_ziyan.head(30)


# missing values in the 'body' column with the string "missing"
import pandas as pd
data_ziyan=pd.read_csv('C:/Users/14129/Desktop/dataWarehs/titanic.csv')
data_ziyan['body'].head(30)

##
data_ziyan['body'].fillna("missing",inplace=True)
data_ziyan['body'].head(30)



# use the average to fill in the missing age
import pandas as pd
data_ziyan=pd.read_csv('C:/Users/14129/Desktop/dataWarehs/titanic.csv')
data_ziyan['age'].head(30)


# Replace missing values in the 'age' column with the mean age
## get the age mean
print(data_ziyan['age'].mean())
##
data_ziyan['age'].fillna(data_ziyan['age'].mean(),inplace=True)
data_ziyan['age'].head(30)


'''----------------------------------------'''
'''Q3----------------------------------'''

import pandas as pd

data_ziyan=pd.read_csv('C:/Users/14129/Desktop/dataWarehs/titanic.csv')
data_ziyan.columns.values

# Creating dummy variables for the 'sex' column
# create dummy dataframe
dummy_sex=pd.get_dummies(data_ziyan['sex'],prefix='sex')
print(dummy_sex.head())
dummy_sex.head()

# Merging the dummy variables back into the original DataFrame
# join the dummy datframe to the original dataset and remove the original column
column_name=data_ziyan.columns.values.tolist()
column_name
column_name.remove('sex')# Removing the original 'sex' column
column_name

# Joining the dummy dataframe with the original dataframe and printing it
data_ziyan[column_name].join(dummy_sex)
print(data_ziyan[column_name].join(dummy_sex))

'''----------------------------------------'''
'''Q4----------------------------------'''
import matplotlib.pyplot as plt

import pandas as pd

data1_ziyan = pd.read_csv('C:/Users/14129/Desktop/dataWarehs/Customer Churn Model.txt')

print(data1_ziyan.columns.values)


# Creating various plots:

#create a scatterplot

fig_ziyan = data1_ziyan.plot(kind='scatter',x='Day Mins',y='Day Charge')

# Save the scatter plot

fig_ziyan.figure.savefig('C:/Users/14129/Desktop/dataWarehs/ScatterPlot.pdf')



figure,axs = plt.subplots(2, 2,sharey=True,sharex=True)

data1_ziyan.plot(kind='scatter',x='Day Mins',y='Day Charge',ax=axs[0][0])

data1_ziyan.plot(kind='scatter',x='Night Mins',y='Night Charge',ax=axs[0][1])

data1_ziyan.plot(kind='scatter',x='Day Calls',y='Day Charge',ax=axs[1][0])

data1_ziyan.plot(kind='scatter',x='Night Calls',y='Night Charge',ax=axs[1][1])

plt.show()



#plot a histrogram

plt.hist(data1_ziyan['Day Calls'],bins=8)

plt.xlabel('Day Calls Value')

plt.ylabel('Frequency')

plt.title('Frequency of Day Calls')

plt.show()



#Plot a histrogram

import matplotlib.pyplot as plt

plt.boxplot(data1_ziyan['Day Calls'])

plt.ylabel('Day Calls')

plt.title('Box Plot of Day Calls')

plt.show()


'''---------------------------------'''
'''Q5-----------------------------------'''
#######################################################

#Sub setting the data slicing and dicing

#######################################################

## Columns

import pandas as pd

import os

path = "C:/Users/14129/Desktop/dataWarehs"

filename = 'Customer Churn Model.txt'

fullpath = os.path.join(path,filename)

data_ziyan = pd.read_csv(fullpath)

print(data_ziyan.columns.values)


# extract one column (i.e. a series)
account_length=data_ziyan['Account Length']
print(account_length.head(50))
type(account_length)

#extract many columns into a new dataframe
subdata_ziyan = data_ziyan[['Account Length','VMail Message','Day Calls']]
subdata_ziyan.head()
type(subdata_ziyan)

# Creating a DataFrame from a list of specific columns
# Create a list of wanted columns
wanted_columns=['Account Length','VMail Message','Day Calls']
subdata_ziyan=data_ziyan[wanted_columns]
print(subdata_ziyan.head())

## Another way useful when many columns
wanted=['Account Length','VMail Message','Day Calls']
column_list=data_ziyan.columns.values.tolist()
sublist=[x for x in column_list if x not in wanted]
subdata=data_ziyan[sublist]
subdata_ziyan.head()

## Rows

#Select the first 50 rows
data_ziyan[:50]

# select 50 rows starting at 25
print(data_ziyan[25:75])# Selecting rows from 25 to 75


# filter the rows that have clocked day Mins to be greater than 350.
sub_data_ziyan=data_ziyan[data_ziyan['Day Mins']>350]
sub_data_ziyan.shape  # Prints the shape (rows, columns) of the filtered DataFrame
sub_data_ziyan # Displays the filtered DataFrame

#filter the rows for which the state is VA:
sub_data_ziyan=data_ziyan[data_ziyan['State']=='VA']
sub_data_ziyan.shape
sub_data_ziyan


# Filtering rows with multiple conditions
#filter the rows that have clocked day Mins to be greater than 150 and the state value is VA

sub_data_ziyan=data_ziyan[(data_ziyan['Day Mins']>150) & (data_ziyan['State']=='VA')]
sub_data_ziyan.shape
sub_data_ziyan[['State','Day Mins']]

## Create a new column for total minutes
data_ziyan['Total Mins']=data_ziyan['Day Mins']+data_ziyan['Eve Mins']+data_ziyan['Night Mins']
data_ziyan['Total Mins'].head()


'''---------------------------------'''
'''--------------Third Part---------------------'''
''''''
'''---------------------------------'''
'''Q1-----------------------------------'''

#Generate one number between 1 and 100

import numpy as np
np.random.randint(1,100)

#Generate a random number between 0 and 1
import numpy as np
np.random.random()

#Define a function to generate several random numbers in a range
def randint_range_ziyan(n,a,b):
    x=[]
    for i in range(n):
        x.append(np.random.randint(a,b))
    return x
list_x= randint_range_ziyan(5,30,70) # Generating 5 random integers between 30 and 70

#d. Generate three random numbers between 0 and 100, which are all multiples of 5
import random
for i in range(3):
    print( random.randrange(0,100,5))

# Select three numbers randomly from a list of numbers
list = [20, 30, 40, 50 ,60, 70, 80, 90]
sampling = random.choices(list, k=3)# Randomly choosing 3 elements

print("sampling with choices method ", sampling)

#Generate a set of random numbers that retain their value, i.e. use the seed option
np.random.seed(1)# Setting seed for reproducibility
for i in range(3):
    print (np.random.random())

#Shuffle a list of 5 numbers
a = [1,2,3,4,5]
print(a)
np.random.shuffle(a)
print(a)
'''---------------------------------'''
'''Q2-----------------------------------'''

import pandas as pd

filepath='C:/Users/14129/Desktop/dataWarehs/lotofdata'

data_final=pd.read_csv(filepath+'/'+'001.csv')
data_final_size=len(data_final)

# Looping through a series of files and concatenating them to the initial DataFrame
for i in range(1,333):
    if i<10:
        filename='0'+'0'+str(i)+'.csv'
    if 10<=i<100:
        filename='0'+str(i)+'.csv'
    if i>=100:
        filename=str(i)+'.csv'

# Constructing the full file path
file=filepath+'/'+filename

data=pd.read_csv(file)

# Incrementing the total size with the number of rows of the current file
data_final_size+=len(data)
data_final=pd.concat([data_final,data],axis=0)

print (data_final_size)

data_final.shape

