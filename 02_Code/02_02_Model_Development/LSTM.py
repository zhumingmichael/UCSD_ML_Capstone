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
