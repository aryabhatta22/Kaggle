# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:28:47 2021

@author: tarun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
Paths
"""
HOME_DIR = os.getcwd()
PATH_TRAIN = HOME_DIR+'\\Data\\train.csv'
PATH_TEST = HOME_DIR+'\\Data\\test.csv'
PATH_SUB = HOME_DIR+'\\Data\\sample_submission.csv'

"""
Dataframes
"""
df_orig_train = pd.read_csv(PATH_TRAIN)
df_orig_test = pd.read_csv(PATH_TEST)
df_orig_sub = pd.read_csv(PATH_SUB)


df_train = df_orig_train.copy()
df_test = df_orig_test.copy()

"""
Headers
"""

headers_train = df_train.columns.values
headers_test = df_test.columns.values

"""
check for null values
"""

def getNullFeatures(df,name="Unknown dataframe"):
    headers = df.columns.values
    headers_null = []
    for header in headers:
        if (df[header].isnull().sum() > 0):
            headers_null.append((header,df[header].isnull().sum))
    if(len(headers_null) == 0):
        print(">> No Null Values found in {0}".format(name))
    else:
        print(">> Following are the null features of {}: \n\n {}".format(name, headers_null))
    return headers_null

headers_null_train = getNullFeatures(df_train,"Training DF")
headers_null_test = getNullFeatures(df_test, "Test DF")

# Observation --> No null values present in the train or test data

"""
check for datatype of each column
"""

def getColumnDatatype(df,headers_df):
    dict_datatype_features ={}
    dict_datatype_count = {}
    
    for header in headers_df:
        d_type = str(df[header].dtype)
        
        if d_type in dict_datatype_features:
            # if type is not a list
            if not isinstance(dict_datatype_features[d_type], list):
                dict_datatype_features[d_type] = [dict_datatype_features[d_type]]
                
            dict_datatype_features[d_type].append(header)
            dict_datatype_count[d_type] = dict_datatype_count[d_type] + 1
        else:
            dict_datatype_features[d_type] =  header
            dict_datatype_count[d_type] = 1
    
    return dict_datatype_features, dict_datatype_count

dict_train_datatype_features, dict_train_datatype_count = getColumnDatatype(df_train, headers_train)
dict_test_datatype_features, dict_test_datatype_count = getColumnDatatype(df_test, headers_test)

print("\n>> Datatype count for Training columns : \n\n {}".format(dict_train_datatype_count))
print("\n>> Datatype count for Test columns : \n\n {}".format(dict_test_datatype_count))

print("\n>>The only object type column in training set is '{}'. ".format(dict_train_datatype_features['object'] ))

# Observation --> Hence only label id to be encoded, rest all the features are of numeric type. This label could be encoded at the time of modeling.

"""
Correlation
"""

sns.heatmap(df_train.corr(), cmap="coolwarm")

# Observation --> All the features are barely correlated


"""
Outliers
"""
Q1 = df_train.quantile(0.25)
Q3 = df_train.quantile(0.75)
IQR = Q3-Q1
print(IQR)

#Lets set all the outliers possibility to No intially
df_train['Has Outlier'] =  'No'
df_test['Has Outlier'] =  'No'

# Function to check whether a value on specific column is an outlier or not
def checkForOutlier(row,header,Q1,Q3,IQR):
    # if the value is beyond inter quratile range
    if ((row[header] < (Q1 - 1.5 * IQR)) or (row[header] > (Q3 + 1.5 * IQR))):
        return "Yes"
    #if row has not already been set to Yes for an earlier outlier value
    elif(row['Has Outlier'] != "Yes"):
        return "No"
        
# check outliers for training data
for key, value in IQR.items():
    if(str(key) != 'id'):
        df_train['Has Outlier'] = df_train.apply(checkForOutlier, args=(str(key),Q1[key],Q3[key], value,), axis=1)
        
# check outliers for testing data with same IQR
for key, value in IQR.items():
    if(str(key) != 'id'):
        df_test['Has Outlier'] = df_test.apply(checkForOutlier, args=(str(key),Q1[key],Q3[key], value,), axis=1)
    

# variables to hold total numbers of outliers in both the training and test set
total_outliers_train = df_train['Has Outlier'].value_counts()['Yes']
total_outliers_test = df_test['Has Outlier'].value_counts()['Yes']

print("Number of Outliers in Training Data ", total_outliers_train)
print("Total Outliers in training data {}% ".format(total_outliers_train*100/df_train.shape[0]))

print("Number of Outliers in Training Data ", total_outliers_test)
print("Total Outliers in training data {}% ".format(total_outliers_test*100/df_test.shape[0]))

# Observation --> Since there are >19% of rows which have one or more outlier thus we will leave the outliers for now
#                   and make any chnages after testing our model performance

