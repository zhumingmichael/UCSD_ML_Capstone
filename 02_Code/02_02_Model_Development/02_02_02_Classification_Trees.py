%reset -f

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix,\
    f1_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import re

sMyPath = r'C:\UCSD_ML_Capstone'
os.chdir(sMyPath)


#### Import Data
##### Feature Generation #####
### Return the file names
# from os import listdir
# from os.path import isfile, join

datapath = sMyPath + r'\01_Input\01_03_ProcessedData\Tiingo_Stock_daily_Transformed'
# AvaTickers_df = pd.read_csv(sMyPath + r"\01_Input\01_01_DataCodes\Stock_Techonology_Semiconductors.csv")
# AvaTickers  = AvaTickers_df..values
AvaTickers_df = pd.read_csv(sMyPath + r"\01_Input\01_01_DataCodes\sNA_Select_Tickers.csv")
AvaTickers = AvaTickers_df[(AvaTickers_df.Sector=='Technology')&(AvaTickers_df.Industry=='Semiconductors')].Tickers.values



#### 1. Data clean ####
Ticker = 'RMBS'

### Remove features with only 1 unque value
DataRaw = pd.read_pickle(datapath + r'\\'+ Ticker + r'_tsfresh.zip')
DataRaw = DataRaw.loc[:,[not bool(re.search('duplicate', x)) for x in DataRaw.columns]]

#####



[bool(re.search('has_duplicate'))]























