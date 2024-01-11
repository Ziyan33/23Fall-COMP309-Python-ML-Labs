# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:51:30 2023

@author: 14129
"""

# Import necessary libraries
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define the path and filename for your data_group3set
path = "C:/Users/14129/Desktop/dataWarehs"
filename = 'Bicycle_Thefts_Open_Data.csv'
fullpath = os.path.join(path, filename)

# Read the data_group3set
data= pd.read_csv(fullpath)


# Display the first few rows of the dataframe
print("First Few Rows of the Dataframe:")
print(data.head())

# Summary of the dataset
print("\nSummary Statistics of the Dataset:")
print(data.describe(include='all'))

# Information about the dataset
print("\nInformation about the Dataset:")
print(data.info())

# Check for missing values in each column
print("\nMissing Values in Each Column:")
missing_values = data.isnull().sum()
print(missing_values)

print(data["STATUS"].unique())
"""
///////////////////////////////////////////////////////////////
"""
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Handling Missing Data
# For numerical columns, fill missing values with the mean
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].apply(lambda x: x.fillna(x.mean()))

# For categorical columns, fill missing values with the mode (most frequent value)
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Categorical Data Management
# Convert categorical columns using Label Encoding
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Data Normalization/Standardization
# Apply standardization to numerical columns
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

"""
///////////////////////////////////////////////////////////////
"""
# Assuming 'STATUS' is the target variable
# Encoding the target variable if it's categorical
if data['STATUS'].dtype == 'object':
    data['STATUS'] = label_encoder.fit_transform(data['STATUS'])

# Selecting the chosen features and the target variable
X = data[['PRIMARY_OFFENCE', 'BIKE_COST', 'REPORT_DOY', 'BIKE_MAKE', 'BIKE_MODEL']]
y = data['STATUS']


"""
///////////////////////////////////////////////////////////////
"""


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for class imbalance in the target variable
print(y_train.value_counts())

# If imbalanced, use SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Checking the class balance after applying SMOTE
print(y_train_smote.value_counts())



"""
///////////////////////////////////////////////////////////////
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the models
log_reg = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier()

# Train the models
log_reg.fit(X_train_smote, y_train_smote)
decision_tree.fit(X_train_smote, y_train_smote)

# Predictions on the test set
y_pred_log_reg = log_reg.predict(X_test)
y_pred_decision_tree = decision_tree.predict(X_test)

# Evaluating Logistic Regression Model
print("Logistic Regression Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Classification Report:")
print(classification_report(y_test, y_pred_log_reg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))

# Evaluating Decision Tree Model
print("\nDecision Tree Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_decision_tree))
print("Classification Report:")
print(classification_report(y_test, y_pred_decision_tree))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_decision_tree))



import pickle


# Save the model to a file
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(decision_tree, file)
# Assuming your Logistic Regression model is named 'log_reg'
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(log_reg, file)

"""
///////////////////////////////////////////////////////////////
"""



import pickle
from sklearn.preprocessing import LabelEncoder

# Assuming 'bike_make' and 'bike_model' are categorical columns in your dataset
bike_make_encoder = LabelEncoder()
data['bike_make_encoded'] = bike_make_encoder.fit_transform(data['BIKE_MAKE'])

bike_model_encoder = LabelEncoder()
data['bike_model_encoded'] = bike_model_encoder.fit_transform(data['BIKE_MODEL'])

# Save the encoders to pickle files
with open('bike_make_encoder.pkl', 'wb') as file:
    pickle.dump(bike_make_encoder, file)

with open('bike_model_encoder.pkl', 'wb') as file:
    pickle.dump(bike_model_encoder, file)














