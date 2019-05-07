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

class_num = 3

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
        Y_true = np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["Close"])
        Y_last = np.array(train.iloc[i+pastDay-1:i+pastDay]["Close"])
        if (((Y_true - Y_last) / Y_last *100) >= 2): 
            Y_train.append(oneHot(0, class_num))
        elif(((Y_true - Y_last) / Y_last *100) >= -2):
            Y_train.append(oneHot(1, class_num))
        else:
            Y_train.append(oneHot(2, class_num))
        #Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["Close"]))
    return np.array(X_train), np.array(Y_train)
def buildMultiTrain(X_train, Y_train, train, pastDay=30, futureDay=5):
    x , y = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        x.append(np.array(train.iloc[i:i+pastDay]))        
        Y_true = np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["Close"])
        Y_last = np.array(train.iloc[i+pastDay-1:i+pastDay]["Close"])
        if (((Y_true - Y_last) / Y_last *100) >= 2): 
            y.append(oneHot(0, class_num))
        elif(((Y_true - Y_last) / Y_last *100) >= -2):
            y.append(oneHot(1, class_num))
        else:
            y.append(oneHot(2, class_num))
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
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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

source = ['INTC', 'AMD', 'CSCO','AAPL','MU','NVDA','QCOM','AMZN','NFLX','FB','EBAY','IBM','XLNX','TXN','NOK','TSLA','MSFT','SNPS']
for i in range (len(source)):
    train = readTrain(source[i] + ".csv")

# Augment the features (year, month, date, day)
    train_Aug = augFeatures(train)

# Normalization
    train_norm = normalize(train_Aug)
    if i == 0:
        X_train, Y_train = buildTrain(train_norm, 30, 1)
# build Data, use last 30 days to predict next 5 days
    else:
        X_train, Y_train = buildMultiTrain(X_train, Y_train, train_norm, 30, 1)

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


model.save('model/model_double_label' + '.h5')

#plt.plot(X_train["Adj Close"])
plt.plot(model.predict(X)[0:100])
plt.plot(Y[0:100])
#plt.plot(Y)