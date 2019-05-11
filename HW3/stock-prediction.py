# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:50:32 2019

@author:Jason
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

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
    global scl
    train = train.drop(["Date"], axis=1)
    print(np.mean(train))
    #train_norm = scl.fit_transform(train)
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm
def buildTrain(train, pastDay=30, futureDay=5):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["Close"]))
    return np.array(X_train), np.array(Y_train)
def buildMultiTrain(X_train, Y_train, train, pastDay=30, futureDay=5):
    x , y = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        x.append(np.array(train.iloc[i:i+pastDay]))        
        y.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["Close"]))
    print (np.shape(X_train))
    print (np.shape(x))
    X_train = np.concatenate((X_train, x), axis=0)
    Y_train = np.concatenate((Y_train, y), axis=0)
    return np.array(X_train), np.array(Y_train)
def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]
def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val
def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
  # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model
def output_result(history):
    '''
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()
    '''
    # summarize history for loss 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss']) 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()

source = ['INTC', 'AMD', 'CSCO','AAPL','MU','NVDA','QCOM','AMZN','NFLX','FB','GOOG','BABA','EBAY','IBM','XLNX','TXN','NOK','TSLA','MSFT','SNPS']
for i in range (len(source)):
    train = readTrain(source[i] + ".csv")

# Augment the features (year, month, date, day)
    train_Aug = augFeatures(train)

# Normalization
    train_norm = normalize(train_Aug)
    if i == 0:
        X_train, Y_train = buildTrain(train_norm, 60, 1)
# build Data, use last 30 days to predict next 5 days
    else:
        X_train, Y_train = buildMultiTrain(X_train, Y_train, train_norm, 60, 1)

X = X_train
Y = Y_train
# shuffle the data, and random seed is 10
X_train, Y_train = shuffle(X_train, Y_train)

# split training data and validation data
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)
model = buildManyToOneModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
history = model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
output_result(history)


model.save('model/model_10' + '.h5')

#plt.plot(X_train["Adj Close"])
plt.plot(model.predict(X))
plt.plot(Y)
#plt.plot(Y)