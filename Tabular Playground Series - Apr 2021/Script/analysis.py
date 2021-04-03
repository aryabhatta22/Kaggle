# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 20:44:13 2021

@author: tarun
"""

#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


#Global Variables


path_train_file = '../Data/train.csv'
path_test_file = '../Data/test.csv'



#Reading File and deatils

df_train = pd.read_csv(path_train_file)
df_test = pd.read_csv(path_test_file)

#headers

headers_train = df_train.columns.values
headers_test = df_test.columns.values
headers_train_null =  []
headers_test_null =  []
headers_irrelavant = ['PassengerId','Ticket','Fare']


def findNullColumns(df, headers):
    headers_null =[]
    for header in headers:
        if df[header].isnull().any():
            headers_null.append((header,df[header].isnull().sum()))
    return headers_null

headers_train_null = findNullColumns(df_train, headers_train)        
headers_test_null = findNullColumns(df_test, headers_test)
        
        
"""
1. Removing irrelavnt columns

PassengerID,ticket, fare

"""

df_train = df_train.drop(columns=headers_irrelavant)
df_test = df_test.drop(columns=headers_irrelavant)


"""
2. Handling missing values

age,cabin.embarked

"""

        #Age

#number of missing values is 3292

# Histogram to detect any skewed distribution

fig, ax = plt.subplots(2, figsize = (7,5))
fig.suptitle("Histogram of Age")
ax[0].set_title('Training Data')
sns.histplot(df_train['Age'], kde=True,bins=20, ax=ax[0])
ax[1].set_title('Test Data')
sns.histplot(df_test['Age'], kde=True,bins=20, ax=ax[1])
fig.tight_layout()
plt.show()

# Boxplot to detect any outlier

fig, ax = plt.subplots(2, figsize = (8,8))
fig.suptitle("Boxplot of Age")
ax[0].set_title('Training Data')
sns.boxplot(df_train['Age'], ax=ax[0])
ax[1].set_title('Test Data')
sns.boxplot(df_test['Age'], ax=ax[1])
fig.tight_layout()
plt.show()


# Mean, median, mode
print("---> Age (Training Data)")
print('Mean: ', df_train['Age'].mean())
print('Median: ', df_train['Age'].median())
print('Mode: ', df_train['Age'].mode())
print('Std: ', df_train['Age'].std())

print("\n---> Age (Test Data)")
print('Mean: ', df_test['Age'].mean())
print('Median: ', df_test['Age'].median())
print('Mode: ', df_test['Age'].mode())
print('Std: ', df_test['Age'].std())


# Replacing All the null values of training with mean of the data
# and median for the test data due to presence of outliers

df_train['Age'].fillna( df_train['Age'].mean(), inplace = True)
df_test['Age'].fillna( df_test['Age'].median(), inplace = True)

