%reset -f

import numpy as np
import pandas as pd
import tensorflow as tf
import os

from datetime import datetime as DT
from datetime import timedelta
from copy import deepcopy as DC
import matplotlib.pylab as plt

# Let's load the libraries and dependencies for the deep learning model
from sklearn.preprocessing import MinMaxScaler

# %tensorflow_version 1.x
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD

sMyPath = r'C:\UCSD_ML_Capstone'
datapath = sMyPath + r'\01_Input\01_03_ProcessedData\Tiingo_Stock_daily_Transformed'

Ret_D = pd.read_pickle(datapath + r"\Ret_D.zip")
Price_log = pd.read_pickle(datapath + r"\Price_log.zip")

df = pd.concat([Ret_D,Price_log], axis=1)

##33#
train = df[:"2019"].values.copy()
test = df["2020":].values.copy()

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=64, batch_size=72,\
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
