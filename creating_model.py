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

def TrainTestSplit(input, output, testSize):
    
    #rng = np.random.default_rng()
    #rand= rng.integers(100)
    rand = 37
    Xtest = input.sample(frac=testSize, random_state=rand)
    Xtrain = input.drop(Xtest.index)
    
    Ytest = output.sample(frac=testSize, random_state=rand)
    Ytrain = output.drop(Ytest.index)
    
    return Xtrain, Xtest, Ytrain, Ytest
    

########## CUSTOM MODEL CODE (this function)##########
def RunModel (model, Xtrain, Xtest, Ytrain, Ytest, sc):
    
    model.fit(Xtrain, Ytrain)    
    predictions = model.predict(Xtest)
    
    scaleUpOutput = False
    if (scaleUpOutput):
        #(doesnt work)
        Ytest = sc.inverse_transform(Ytest)
        predictions = sc.inverse_transform(predictions)[0]
        
    rmse = mean_squared_error(Ytest, predictions, squared=False)
    return rmse


def CustomModel(x_train):
    try:
        customModel = load("custom_model.joblib")
        return customModel
    except:
        #creating the model and adding layers to it
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=32, activation='relu', input_dim=len(x_train.columns)))
        model.add(keras.layers.Dense(units=64, activation='relu'))
        model.add(keras.layers.Dense(units=64, activation='relu'))
        model.add(keras.layers.Dense(units=64, activation='relu'))
        model.add(keras.layers.Dense(units=64, activation='relu'))
        model.add(keras.layers.Dense(units=1, activation='sigmoid'))

        model.compile(loss="binary_crossentropy", optimizer="sgd")    
        
        dump(model, "custom_model.joblib")
        return model




#import dataset
dataset = pd.read_csv("Car_Purchasing_Data.csv")
 
#creating input data`set by removing irrelevant columns
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

########## CUSTOM MODEL CODE (1 line) ##########
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
########## CUSTOM MODEL CODE (2 lines) ##########
    if modelName is "Sequential":
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


