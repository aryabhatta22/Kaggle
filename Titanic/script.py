# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:37:45 2020

@author: tarun
"""

# ---------------------- Data preprocessing ---------------------- 

import numpy as np
import pandas as pd

df_train = pd.read_csv('train.csv')
df_train.Age.fillna(df_train.Age.median(), inplace=True)
df_test = pd.read_csv('test.csv')
df_test.Age.fillna(df_test.Age.median(), inplace=True)
df_test_y = pd.read_csv('gender_submission.csv')

X_train = df_train.loc[:, ['Pclass','Sex', 'Age','SibSp','Parch'] ].values   
y_train = df_train.loc[:, ['Survived' ] ].values   

#print(df_train.isnull().any())
#print(df_test.isnull().any())

X_test = df_test.loc[:, ['Pclass','Sex', 'Age','SibSp','Parch']].values
y_test = df_test_y.iloc[:,1].values

#Categorical distribution

from sklearn.preprocessing import LabelEncoder

labelencoder_X_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder_X_1.transform(X_test[:, 1])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------------------- Neural Network ---------------------- 

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 5 ))      
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid', input_dim = 5 ))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# ---------------------- Predictions ---------------------- 
from numpy import savetxt
y_pred = classifier.predict(X_test)

for i in range(418):
    if (y_pred[i] > 0.5):
        y_pred[i]=1
    else:
        y_pred[i]=0

savetxt('result.csv', y_pred, delimiter=',')
        

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
