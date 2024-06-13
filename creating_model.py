import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from joblib import load

import tensorflow as tf
import keras

#make the model work well
#remember to upload on the other repo
def RunModel (model, Xtrain, Xtest, Ytrain, Ytest, sc):
    #train and test the model
    model.fit(Xtrain, Ytrain)    
    predictions = model.predict(Xtest)
    
    #turn predictions and Ytest back into normal values
    try:
        inv_predictions = scOutput.inverse_transform(predictions)
    except:
        inv_predictions = scOutput.inverse_transform([predictions])[0]

    inv_Ytest = scOutput.inverse_transform(Ytest)
    
    #make scatter chart for the custom model
    if type(model).__name__ == "Sequential":    
        plt.scatter(inv_Ytest, inv_predictions)
        plt.xlabel("Actual")
        plt.ylabel("Prediction")
        plt.title(type(model).__name__)
        ax = plt.gca()
        ax.set_xlim([0, 100000])
        ax.set_ylim([0, 100000])
        plt.show()
    
    #get rmse from model's performance
    scaleRMSE = True 
    if scaleRMSE:
        rmse = mean_squared_error(inv_Ytest, inv_predictions, squared=False)
    else:
        rmse = mean_squared_error(Ytest, predictions, squared=False)
        
    return rmse


def CustomModel(x_train):
    #creating the model and adding layers to it
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=320, activation='relu', input_dim=len(x_train.columns)))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=640, activation='relu'))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=640, activation='relu'))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer="adam")    
    
    dump(model, "custom_model.joblib")
    return model




#import dataset
dataset = pd.read_csv("Car_Purchasing_Data.csv")
 
#creating input data set by removing irrelevant columns
#### input => X ####
inputData = dataset.filter(["Age", "Annual Salary", "Credit Card Debt", "Net Worth"])

#creating output dataset
#### output => Y ####
outputData = dataset.filter(["Car Purchase Amount"])

#scale input and output
scInput = MinMaxScaler()
inputScaled = scInput.fit_transform(inputData)
inputScaled = pd.DataFrame(inputScaled)

scOutput = MinMaxScaler()
outputReshaped = outputData.values.reshape(-1,1)
outputScaled = scOutput.fit_transform(outputReshaped)
outputScaled = pd.DataFrame(outputScaled)

#separate into training data and testing data
inputTrain, inputTest, outputTrain, outputTest = train_test_split(inputScaled, outputScaled, test_size=0.2, random_state=37)






#initializing 10 models + custom
modelList = []
rmseList = []
modelNameList = []

modelList.append(CustomModel(inputTrain))
modelList.append(linear_model.LinearRegression())
modelList.append(linear_model.Ridge())
modelList.append(linear_model.Lasso())
modelList.append(linear_model.ElasticNet())
modelList.append(linear_model.Lars())
modelList.append(linear_model.OrthogonalMatchingPursuit())
modelList.append(linear_model.BayesianRidge())
modelList.append(SVR())
modelList.append(ensemble.RandomForestRegressor())
modelList.append(ensemble.GradientBoostingRegressor())

#training and testing 10 models
for model in modelList:
    rmse = RunModel(model, inputTrain, inputTest, outputTrain, outputTest, scOutput)
    rmseList.append(rmse)
    
    modelName = type(model).__name__
    if modelName == "Sequential":
        modelName = "Custom Model"
    modelNameList.append(modelName)
        
#making the RMSE comparison chart
plt.figure(figsize=(12,6))
bar = plt.bar(modelNameList, rmseList)
plt.bar_label(bar)
plt.ylabel("RMSE")
plt.title("RMSE (Root Mean Squared Error) of Regression Models")
plt.xticks(rotation=25, ha="right")
plt.subplots_adjust(bottom=0.20)
plt.show()



