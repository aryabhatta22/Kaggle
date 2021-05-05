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
headers_irrelavant = ['PassengerId','Name','Ticket','Fare']


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

PassengerID,Name,ticket, fare

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

        # Cabin

# since more than 50% of total values in the training and test data is null, we will drop the cabin column

df_train = df_train.drop(columns=['Cabin'])
df_test = df_test.drop(columns=['Cabin'])


        # Embarked

# get percent of each emabrked class

def getDistribution(df):
    dict_value_count = df.value_counts()
    size_df = df.shape[0]
    for key in dict_value_count.keys():
        dict_value_count[key] = float(dict_value_count[key]*100/size_df)
    print(dict_value_count)
    return

getDistribution(df_train['Embarked'])
getDistribution(df_test['Embarked'])

sns.countplot(df_train['Embarked'], hue=df_train["Survived"])

#since class 'S' has more than 3/4th of the non null values for trainig 
# and similar ratio for test and null values is just ~1% or ~2%, we can replcae it with

df_train['Embarked'].fillna( 'S', inplace = True)
df_test['Embarked'].fillna( 'S', inplace = True)


headers_train_null = findNullColumns(df_train, headers_train)        
headers_test_null = findNullColumns(df_test, headers_test)

"""
3. Check for class imbalance

"""

getDistribution(df_train['Survived'])
sns.countplot(df_train['Survived'])

# thus there's no class imbalance

"""
4. Count Plot

"""
fig, ax = plt.subplots(2,2, figsize=(10,5))
fig.suptitle("Count Plot")

sns.countplot(df_train['Pclass'], hue= df_train['Survived'], ax=ax[0,0])
ax[0,0].set_title('Pclass')

sns.countplot(df_train['Sex'], hue= df_train['Survived'] , ax=ax[0,1])
ax[0,1].set_title('Sex')

sns.countplot(df_train['SibSp'], hue= df_train['Survived'] , ax=ax[1,0])
ax[1,0].set_title('SibSp')

sns.countplot(df_train['Parch'], hue= df_train['Survived'] , ax=ax[1,1])
ax[1,1].set_title('Parch')


fig.tight_layout()
plt.show()

"""
5. Adding New Features

"""

# Age Category

def categorizeAge(row):
    if row['Age'] >= 0 and row['Age'] <=5:
        ageCategory = -1
    elif row['Age'] > 5 and row['Age'] <=18:
        ageCategory = 0
    elif row['Age'] > 18 and row['Age'] <=30:
        ageCategory = 1
    elif row['Age'] > 30 and row['Age'] <=60:
        ageCategory = 2
    elif row['Age'] > 60 and row['Age'] <=80:
        ageCategory = 3
    else:
        ageCategory = 4
    return ageCategory

df_train['Age Category'] = df_train.apply(categorizeAge, axis = 1)
df_test['Age Category'] = df_test.apply(categorizeAge, axis = 1)

# Lone Traveler

def isLoneTraveler(row):
    if row['SibSp'] == 0 and row['Parch'] == 0:
        loneTraveler = 1
    else:
        loneTraveler = 0
    return loneTraveler

df_train['Lone Traveler'] = df_train.apply(isLoneTraveler, axis = 1)
df_test['Lone Traveler'] = df_test.apply(isLoneTraveler, axis = 1)


"""
6. Save Training and Test data

"""

df_train.to_csv("../Data/clean_training.csv", index=False)
df_test.to_csv("../Data/clean_test.csv", index=False)