# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:06:51 2023

@author: 14129
"""

import pandas as pd
import os
path = "C:/Users/14129/Desktop/dataWarehs"
filename = 'Bicycle_Thefts_Open_Data.csv'
fullpath = os.path.join(path,filename)
## read
data_ziyan = pd.read_csv(fullpath)

# Display the first few rows of the dataset for an overview
print(data_ziyan.head())

# Get descriptive statistics of the dataset
print(data_ziyan.describe(include='all'))

# Check data types explicitly
print(data_ziyan.dtypes)

# Check for missing values in each column
print(data_ziyan.isnull().sum())

# Calculate the percentage of missing data in each column
missing_percentage = (data_ziyan.isnull().sum() / len(data_ziyan)) * 100
print("Percentage of missing data in each column:\n", missing_percentage)

# Get unique values in the 'STATUS' column
unique_statuses = data_ziyan['STATUS'].unique()

# Print the unique values
for status in unique_statuses:
    print(status)

"""

-----------------------------------------------------
"""
#Bike Color and Make: Some bike types or colors can be easier to sell.
import matplotlib.pyplot as plt

# Count the occurrences of each bike color
color_counts = data_ziyan['BIKE_COLOUR'].value_counts()

# Plotting the top 10 bike colors involved in thefts
color_counts.head(10).plot(kind='bar')
plt.title('Top 10 Bike Colors in Thefts')
plt.xlabel('Bike Color')
plt.ylabel('Number of Thefts')
plt.show()

# Count the occurrences of each bike make
make_counts = data_ziyan['BIKE_MAKE'].value_counts()

# Plotting the top 10 bike makes involved in thefts
make_counts.head(10).plot(kind='bar')
plt.title('Top 10 Bike Makes in Thefts')
plt.xlabel('Bike Make')
plt.ylabel('Number of Thefts')
plt.show()
#Most stolen bike color is black, least stole bike is WHIPLE color

"""
-----------------------------------------------------
"""
#Heat Map
import folium
from folium.plugins import HeatMap

# Filter out rows where coordinates are missing
data_filtered = data_ziyan.dropna(subset=['LAT_WGS84', 'LONG_WGS84'])

# Create a map centered around an average location
map_center = [data_filtered['LAT_WGS84'].mean(), data_filtered['LONG_WGS84'].mean()]
map = folium.Map(location=map_center, zoom_start=12)

# Create a heat map layer
heat_data = [[row['LAT_WGS84'], row['LONG_WGS84']] for index, row in data_filtered.iterrows()]
HeatMap(heat_data).add_to(map)

# Save the map to an HTML file
map.save('bicycle_thefts_heatmap.html')

'''-----------------------------------------------'''


# Create a correlation matrix

# Select only the numerical columns from the dataset
numerical_data = data_ziyan.select_dtypes(include=['int64', 'float64'])

# Calculate the correlation matrix
correlation_matrix = numerical_data.corr()
print(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
