# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 23:27:30 2021

@author: tarun
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


"""
 Model Training
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier


# K nearest neighbors
clf_knn = KNeighborsClassifier(n_neighbors = 5, metric ='minkowski', p =2)
clf_knn.fit(X_train, y_train)

y_pred_knn = clf_knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
ac_knn = accuracy_score(y_test,y_pred_knn)


# random forest
clf_rf = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=0)
clf_rf.fit(X_train,y_train)

y_pred_rf = clf_rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
ac_rf = accuracy_score(y_test,y_pred_rf)

# Logistic Regression
clf_lr = LogisticRegression(random_state = 0)
clf_lr.fit(X_train, y_train)

y_pred_lr = clf_lr.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)
ac_lr = accuracy_score(y_test,y_pred_lr)

# Kernel SVM
clf_svm = SVC(kernel = 'rbf', random_state = 0)
clf_svm.fit(X_train, y_train)

y_pred_svm = clf_svm.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
ac_svm = accuracy_score(y_test,y_pred_svm)

# Naive Bayes
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)

y_pred_nb = clf_nb.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)
ac_nb = accuracy_score(y_test,y_pred_nb)

# Support Vector Machine
clf_svc = SVC(kernel = 'linear', random_state = 0)
clf_svc.fit(X_train, y_train)

y_pred_svc = clf_svc.predict(X_test)
cm_svc = confusion_matrix(y_test, y_pred_svc)
ac_svc = accuracy_score(y_test,y_pred_svc)

# ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

clf_ann = Sequential()                   
clf_ann.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 9 ))                                                 
clf_ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid', input_dim = 11 ))
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
history = clf_ann.fit(X_train, y_train, epochs = 20, batch_size = 32, shuffle = False, validation_data=(X_test, y_test))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Performace')
plt.ylabel('Acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Performace')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

y_pred_ann = clf_ann.predict(X_test)
y_pred_ann = np.where(y_pred_ann >= 0.5, 1,0)
cm_ann = confusion_matrix(y_test, y_pred_ann)
ac_ann = accuracy_score(y_test,y_pred_ann)

# XGBoost Classifier
clf_xgb= XGBClassifier(n_estimators = 150, max_depth  = 8,learning_rate = 0.001,
                       verbosity =1,subsample=0.6,min_child_weight=10,gamma =5,
                       colsample_bytree=1.0)
clf_xgb.fit(X_train, y_train)

y_pred_xgb = clf_xgb.predict(X_test)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
ac_xgb = accuracy_score(y_test,y_pred_xgb)

#GridSeach
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

params = {
        'n_estimators': [10,25,50,100,150,175,200,225],
        'max_depth': [5,8,10,20],
        'min_child_weight': [1, 5, 10,20,30],
        'gamma': [0.5, 1, 1.5, 2, 5,7.5,8,10],
        'subsample': [0.2,0.3,0.6],
        'colsample_bytree': [0.6, 0.8, 1.0,1.2,1.5],
        }

clf_xgb= XGBClassifier( learning_rate = 0.001,
                       verbosity =1 )
folds = 5
param_comb = 6

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 0)
random_search = RandomizedSearchCV(clf_xgb, param_distributions=params,
                                   n_iter=param_comb, scoring='accuracy',
                                   n_jobs=-1, cv=skf.split(X,y), verbose=3, random_state=0 )

print('Random Search Started  ...........\n\n')
start = time.process_time()
random_search.fit(X, y)
print('Random Search Finsihed!!! Took '+str(((time.process_time() - start)/60))+" minutes\n\n")

print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)

"""
 Print Performance
"""
from prettytable import PrettyTable

table_ = PrettyTable()
table_.field_names = ['Model', "Accuracy"]
table_.add_row(["KNN","{:.2f}".format(ac_knn)])
table_.add_row(["Random Forest","{:.2f}".format(ac_rf)])
table_.add_row(["Logistic Regression","{:.2f}".format(ac_lr)])
table_.add_row(["Kernel SVM","{:.2f}".format(ac_svm)])
table_.add_row(["Naive Bayes","{:.2f}".format(ac_nb)])
table_.add_row(["SVM","{:.2f}".format(ac_svc)])
table_.add_row(["ANN","{:.2f}".format(ac_ann)])

"""
 Creating Submission file
"""

df_sub  = pd.read_csv('../Data/sample_submission.csv')
pessenger_id = df_sub.loc[:,'PassengerId'].values
survive_pred = clf_xgb.predict(sub_X)
survive_pred = np.where(survive_pred >= 0.5, 1,0)

Submission_mat = np.concatenate((pessenger_id.reshape(len(pessenger_id),1),
                                 survive_pred.reshape(len(survive_pred),1)), axis = 1)
df_submission = pd.DataFrame(Submission_mat, columns=df_sub.columns.values)

df_submission.to_csv("../Data/submission_XGB.csv", index=False)