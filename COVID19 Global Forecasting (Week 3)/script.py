import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df_train = pd.read_csv('train.csv')
df_train['Province_State'].fillna("",inplace = True)        # for avoiding error during conctination
df_test = pd.read_csv('test.csv')
df_test['Province_State'].fillna("",inplace = True)
df_sub = pd.read_csv('submission.csv')
                
                                    # Data Preprocessing
                
X = df_train.iloc[:, [1,2,3]].values
y = df_train.iloc[:, [4,5]].values
X_test = df_test.iloc[:, [1,2,3]].values
X_sub = df_sub.iloc[:,0:4].values

X[:,1] = X[:,0]+" "+X[:,1]      # to concatinate province and country so that different model could be created for different region
X = X[:, [1,2]]                 # to consider country(province + country) & date only
X_test[:,1] = X_test[:,0]+" "+X_test[:,1]
X_test = X_test[:,[1,2]]

lengthX = len(X)
X = np.append(X, X_test, axis= 0)       # So that label encoder could match in both set


from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])     
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:,1])

X_test[:, 0] = labelencoder_X_1.transform(X_test[:, 0]) 
X_test[:, 1] = labelencoder_X_2.transform(X_test[:,1])

X= X[0:lengthX,:]                   # getting original training set
y = y[X[:,0].argsort()]             # to arrange data in ascending order of label encoding
X = X[X[:,0].argsort()]
X_sub = X_sub[X_test[:,0].argsort()]
X_test = X_test[X_test[:,0].argsort()]


                                    # UPDATE SUBMISSION file
                                    
def updateSubmission(regressor, countryCode,startIndexTest,label):
    if label == "ConfirmedCases":
        j = 1
    elif label == "Fatalities":
        j =2
    X_country = X_test[X_test[:,0] == countryCode] 
    for i in range(0, len(X_country)):
        X_sub[i+startIndexTest,j] = math.floor(regressor.predict(np.asarray(X_country[i,1]).reshape(1,1)))

                                    # CONFIRMED CASE/ FATALITIES PREDICTION

def ConfirmedCasePrediction(countryCode,startingIndex, startIndexTest,label):
    if label == "ConfirmedCases":
        j = 0
    elif label == "Fatalities":
        j =1
    
    i=startingIndex
    X_country = X[X[:,0] == countryCode]                          # X for particular country
    y_country = y[i:i+len(X_country)]                               # y for selected country 
    
    print(label+" in "+ str((labelencoder_X_1.inverse_transform([countryCode]))[0]) +" min "+str (min(y_country[:,0]))+ " max " +str( max(y_country[:,0])))

    
                                        # Random Forest                                     
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
    regressor.fit(X_country[:,1].reshape(len(X_country), 1),y_country[:,j]) 
    updateSubmission(regressor, countryCode,startIndexTest,label)
                        # visulaizing data 
    """
    # Avoid if you dont you just want output not graphs
    X_grid = np.arange(min(X_country[:,1]), max(X_country[:,1]), 0.01)
    X_grid = X_grid.reshape((len(X_grid)), 1) 
    plt.ion()
    plt.figure()
    plt.scatter(X_country[:,1], y_country[:,j], color ='red', label='Confirmed Case')
    plt.plot(X_grid, regressor.predict(X_grid), color='blue',label='Prediction line')
    plt.legend(loc="upper left")
    plt.title('Day vs Confirmed cases in '+ str((labelencoder_X_1.inverse_transform([countryCode]))[0]))
    plt.xlabel("Day")
    plt.ylabel('Confirmed Cases')
    plt.show() 
    """
    
# for predicting Fatalities
startIndex =0
startIndexTest= 0
for i in range(0,len(np.unique(X[:,0]))):
    ConfirmedCasePrediction(i,startIndex,startIndexTest,"Fatalities")
    startIndex = startIndex+ len(X[X[:,0] == i])
    startIndexTest = startIndexTest+(len(X_test[X_test[:,0] == i]))

# for predicting confirmed cases
startIndexTest= 0
startIndex = 0
for i in range(0,len(np.unique(X[:,0]))):
    ConfirmedCasePrediction(i,startIndex,startIndexTest,"ConfirmedCases")
    startIndex = startIndex+ len(X[X[:,0] == i])
    startIndexTest = startIndexTest+(len(X_test[X_test[:,0] == i]))
  
"""
  
# For kaggle submission only
  
my_submission = pd.DataFrame({'ForecastId': X_sub[:,0], 'ConfirmedCases': X_sub[:,1], 'Fatalities': X_sub[:,2]})
my_submission.to_csv('submission.csv', index=False)

"""