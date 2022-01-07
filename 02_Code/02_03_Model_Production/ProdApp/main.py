# %reset -f
from flask import Flask, jsonify, request

import os
import numpy as np
import pandas as pd
import re

## tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import math
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from tsfresh.utilities.dataframe_functions import roll_time_series

import requests
import datetime as dt
from datetime import datetime as DT
from datetime import timedelta
from itertools import chain

sMyPath = r"C:\UCSD_ML_Capstone\02_Code\02_03_Model_Production\ProdApp"

######## 0. Initiate ########
#### 0.1 initiate flask

app = Flask(__name__)

#### 0.2 Ticker range
AvaTickers = np.array(['ADI', 'AMD', 'AMKR', 'ENTG', 'FORM', 'FSLR', 'INTC', 'IPGP',
                       'KLIC', 'LSCC', 'MCHP', 'MPWR', 'MRVL', 'MU', 'OLED', 'ONTO',
                       'POWI', 'RMBS', 'SIMO', 'SLAB', 'SMTC', 'SPWR', 'STM', 'SWKS',
                       'TSEM', 'TSM', 'TXN', 'UCTT', 'UMC', 'XLNX', 'XPER'])

TickerArray = ', '.join(AvaTickers)


#### 0.3 Model import
## initial setting
window_size = 5*26
epochs = 100
batch_size = window_size
# save_dir = sMyPath + r'\ModelProd'
save_dir = r'C:\UCSD_ML_Capstone\02_Code\02_02_Model_Development\LSTM_Model_Save'

iTicker = 0
SNum = np.array(range(len(AvaTickers)))
model_dict = dict()
for iTicker in SNum:
    Ticker = AvaTickers[iTicker]
    model_dict[Ticker] = load_model(save_dir + r"\\" + Ticker + r"_Prod_e" + str(epochs) + r".h5")
    print(r"Import model " + Ticker + r" (" + str(iTicker+1) + r"/" + str(len(SNum)) + r")" )


#### 0.4 Data import setting
P_df = pd.read_csv(sMyPath + r"\sample_data.csv", index_col='Unnamed: 0')


######## 1. API setup ########
@app.route('/')
def Welcome():
    
    Welcome_Lyric = jsonify(message =
                            "Welcome to the stock price prediction API \\n \
                             Please use /parameters to set the request parameters \\n \
                             Tickers: Stock Ticker \\n \
                             Date: As of date of the stock price (format = YYYY-MM-DD) \\n \
                             Ticker should be in the scope of \
                            " +TickerArray )
    return Welcome_Lyric


@app.route('/parameters')
def parameters():
    Ticker = str(request.args.get('Ticker'))

    if Ticker not in AvaTickers:
        return jsonify(message = Ticker + " is not available for now. Please select one ticker from: " + TickerArray), 402
    else: 

        #### Modify the data ####
        P = P_df.tail(window_size-1)
        model = model_dict[Ticker]
    
        P_base = P.iloc[0,:]
        P = P/P_base-1
        X = np.array(P)
        Pred_y = model.predict(X[newaxis,:,:])[0,0]
        Pred_y = (1+Pred_y) * P_base[Ticker]
        
        return jsonify(message = Ticker + r"'s prediction is " + str(round(Pred_y,2) )), 200
        

if __name__ == '__main__':
    app.run()
