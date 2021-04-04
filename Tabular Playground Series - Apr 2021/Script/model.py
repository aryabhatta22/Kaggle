# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 23:27:30 2021

@author: tarun
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

df_train = pd.read_csv('../Data/clean_training.csv')
df_test = pd.read_csv('../Data/clean_test.csv')

def preProcessData(df, isTrain = False):
    # for categorical values
    
    pclass = df.loc[:,'Pclass'].values
    sex = df.loc[:,'Sex'].values
    embarked = df.loc[:,'Embarked'].values
    
    # numeric values
    X = df[['Age','SibSp','Parch']].to_numpy()
    y=[]
    if isTrain:
        y = df_train.loc[:,'Survived'].values
    
    """
    One Hot encoding
    """
    
    #sex
    label_sex = LabelEncoder()
    sex = label_sex.fit_transform(sex)
    sex = to_categorical(sex)
    sex=sex[:,:-1]
    
    # Pclass
    pclass = to_categorical(pclass)
    pclass=pclass[:,:-1]
    
    # Embarked
    label_emb = LabelEncoder()
    embarked = label_emb.fit_transform(embarked)
    embarked = to_categorical(embarked)
    embarked = embarked[:,:-1]
    
    """
    Concatenate all features
    """
    X = np.concatenate((sex, pclass, embarked, X), axis = 1)
    return X,y

"""
    Preprocessing files
"""
X,y = preProcessData(df_train, True)
sub_X, _ = preProcessData(df_test, False)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


"""
 Model Training
"""

# model
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train,y_train)

# evaluation
y_pred = clf.predict(X_test)
cf = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)

"""
 Creating Submission file
"""

df_sub  = pd.read_csv('../Data/sample_submission.csv')
pessenger_id = df_sub.loc[:,'PassengerId'].values
survive_pred = clf.predict(sub_X)

Submission_mat = np.concatenate((pessenger_id.reshape(len(pessenger_id),1),
                                 survive_pred.reshape(len(survive_pred),1)), axis = 1)
df_submission = pd.DataFrame(Submission_mat, columns=df_sub.columns.values)

df_submission.to_csv("../Data/submission_RF_dept10.csv", index=False)