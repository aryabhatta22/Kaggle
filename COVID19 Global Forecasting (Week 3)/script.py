import numpy as np
import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_sub = pd.read_csv('submission.csv')
                
                                    # Data Preprocessing
                
X = df_train.iloc[:, [2,3]].values
y = df_train.iloc[:, [4,5]].values

#print(df_train.isnull().any())
#print(df_test.isnull().any())

from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])     
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:,1])

#temp = X.astype(int)
#temp = temp[y[:,0].sort()]  

                                    # CONFIRMED CASE PREDICTION

def ConfirmedCasePrediction(countryCode,startingIndex):
    import matplotlib.pyplot as plt
    i=startingIndex
    X_country = X[X[:,0] == countryCode]                          # X for particular country
    y_country = y[i:i+len(X_country)]                   # y for selected country 
    y_country = y_country[X_country[:,1].argsort()]
    X_country = X_country[X_country[:,1].argsort()]
    
    
                                        # Random Forest                                     
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
    regressor.fit(X_country[:,1].reshape(len(X_country), 1),y_country[:,0]) 
                        # visulaizing data 
    X_grid = np.arange(min(X_country[:,1]), max(X_country[:,1]), 0.01)
    X_grid = X_grid.reshape((len(X_grid)), 1) 
    plt.ion()
    plt.figure()
    plt.scatter(X_country[:,1], y_country[:,0], color ='red', label='Confirmed Case')
    plt.plot(X_grid, regressor.predict(X_grid), color='blue',label='Prediction line')
    plt.legend(loc="upper left")
    plt.title('Day vs Confirmed cases in '+ str((labelencoder_X_1.inverse_transform([countryCode]))[0]))
    plt.xlabel("Day")
    plt.ylabel('Confirmed Cases')
    plt.show() 
    plt.close('plt')
                        # preydictions
    #import math
    #y_pred = regressor.predict(X_country[:,1].reshape(len(X_country),1))
    #y_pred = [math.floor(case) for case in y_pred]
    
import time

startIndex =0
for i in range(0,len(np.unique(X[:,0]))):
    ConfirmedCasePrediction(i,startIndex)
    time.sleep(0.5)
    startIndex = len(X[X[:,0] == i])