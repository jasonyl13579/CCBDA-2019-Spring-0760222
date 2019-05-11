# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:54:23 2019

@author: Jason
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import load_model

adj_div = 0
adj_mean = 0
class_num = 3
model = load_model('model/model_10' + '.h5') 
input_days = 60 
def oneHot(y, num_class=3):
    y_out = np.zeros(num_class)
    y_out[y] = 1
    return y_out
def readTrain(file):
    train = pd.read_csv(file)
    return train
def augFeatures(train):
    train["Date"] = pd.to_datetime(train["Date"])
    train["year"] = train["Date"].dt.year
    train["month"] = train["Date"].dt.month
    train["date"] = train["Date"].dt.day
    train["day"] = train["Date"].dt.dayofweek
    return train
def normalize(train):
    global adj_mean, adj_div
    train = train.drop(["Date"], axis=1)
    print(np.mean(train))
    #train_norm = scl.fit_transform(train)
    adj_mean = np.mean(train)["Close"]
    adj_div = (np.max(train) - np.min(train))["Close"]
    print (adj_mean)
    print (adj_div)
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x) + 0.001))
    return train_norm
def normalizeAndBuildTest(train, input_days=30, predict_days=2):
    global adj_mean, adj_div
    
    train = train.drop(["Date"], axis=1)
    adj_mean = np.mean(train)["Close"]
    adj_div = (np.max(train) - np.min(train))["Close"]
    train = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x) + 0.001))
    X_test = []
    for i in range(predict_days):
        x_test = train.iloc[train.shape[0] - input_days - i: train.shape[0] - i]   
       # train_norm = x_test.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x) + 0.001))
        X_test.append(np.array(x_test))
    Y_true = train.iloc[train.shape[0]-1:train.shape[0]]["Close"]
    return np.array(X_test), np.array(Y_true)
def buildTrain(train, pastDay=30, futureDay=5):
    X_train, Y_train, test = [], [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["Close"]))
        Y_true = np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["Close"])
        Y_last = np.array(train.iloc[i+pastDay-1:i+pastDay]["Close"])
        if (((Y_true - Y_last) / Y_last *100) >= 2): 
            test.append(oneHot(0, class_num))
        elif(((Y_true - Y_last) / Y_last *100) >= -2):
            test.append(oneHot(1, class_num))
        else:
            test.append(oneHot(2, class_num))
    return np.array(X_train), np.array(Y_train), np.array(test)
def inverse_scaler(x, mean, div):
    return x * div + mean

#model.reset_states()
#%% multi-predict

#source = ['SNPS']

source = ['INTC', 'AMD', 'CSCO','AAPL','MU','NVDA','QCOM','AMZN','NFLX','FB','GOOG','BABA','EBAY','IBM','XLNX','TXN','NOK','TSLA','MSFT','SNPS']
predict = []
date = '0510'
for i in range (len(source)):
    test = readTrain("data/" + date + "/" + source[i] + ".csv")
    test_Aug = augFeatures(test)
    test, Y_true = normalizeAndBuildTest(test_Aug, input_days, 2)
    Y_predict = np.squeeze(inverse_scaler(model.predict(test),adj_mean, adj_div))
    Y_last = inverse_scaler(test[:,input_days-1	,3],adj_mean, adj_div)
    y_t = inverse_scaler(Y_true ,adj_mean, adj_div)
    # display unknown prediction
    print (Y_predict[0], Y_last[0])
    print((Y_predict[0] - Y_last[0]) / Y_last[0])
    
    if (((Y_predict[0] - Y_last[0]) / Y_last[0] *100) >= 2): 
        print (source[i] + ": 0")
        predict.append(0)
    elif(((Y_predict[0] - Y_last[0]) / Y_last[0] *100) >= -2):
        print (source[i] + ": 1")
        predict.append(1)
    else:
        print (source[i] + ": 2")
        predict.append(2)
    '''
    print((y_t[0] - Y_last[1]) / Y_last[1]*100)
    if (((y_t[0] - Y_last[1]) / Y_last[1] *100) >= 2): 
        print ("Yesterday_True" + ": 0")
    elif(((y_t[0] - Y_last[1]) / Y_last[1] *100) >= -2):
        print ("Yesterday_True" + ": 1")
    else:
        print ("Yesterday_True" + ": 2")
   # print ("Yesterday_True:" + str(Y_true[0]))
    if (((Y_predict[1] - Y_last[1]) / Y_last[1] *100) >= 2): 
        print ("Yesterday_predict" + ": 0\n")
    elif(((Y_predict[1] - Y_last[1]) / Y_last[1] *100) >= -2):
        print ("Yesterday_predict" + ": 1\n")
    else:
        print ("Yesterday_predict" + ": 2\n")
    '''
with open( 'data/' + date + '/' + date + '_0760222.txt', 'w+') as f:
    for x in predict:
        f.write(str(x)+ '\n')

#%% predict one
'''
train = readTrain("AMD.csv")

train_Aug = augFeatures(train)

train_norm = normalize(train_Aug)

X_train, Y_train, test = buildTrain(train_norm, input_days, 1)


Y_true = np.squeeze(inverse_scaler(Y_train, adj_mean, adj_div))
Y = model.predict(X_train)
Y_predict = np.squeeze(inverse_scaler(model.predict(X_train),adj_mean, adj_div))
#print (model.predict(X_train[0:10]))
Y_last = inverse_scaler(X_train[:,input_days-1,3],adj_mean, adj_div)
plt.plot(Y_predict)
plt.plot(Y_true)
Rchange1 = np.zeros(np.shape(Y_predict))
Rchange2 = np.zeros(np.shape(Y_predict))
for i in range (len(Rchange1)):
    if (((Y_predict[i] - Y_last[i]) / Y_last[i] *100) >= 2): 
        Rchange1[i] = 0
    elif(((Y_predict[i] - Y_last[i]) / Y_last[i] *100) >= -2):
        Rchange1[i] = 1
    else:
        Rchange1[i] = 2
    if (((Y_true[i] - Y_last[i]) / Y_last[i] *100) >= 2): 
        Rchange2[i] = 0
    elif(((Y_true[i] - Y_last[i]) / Y_last[i] *100) >= -2):
        Rchange2[i] = 1
    else:
        Rchange2[i] = 2
print (Rchange1 == Rchange2)
print (Rchange1)
print (Rchange2)
#print(((Y_predict - Y_last) / Y_last) *100)
#print((Y_true - Y_last) / Y_last*100)
'''
'''
y_pre = model.predict(X_train)
y_pre =  np.argmax(y_pre, 1)
print (y_pre)
print (np.argmax(test, 1))
'''