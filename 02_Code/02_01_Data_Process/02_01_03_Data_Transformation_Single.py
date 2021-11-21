%reset -f

import os
import numpy as np
import pandas as pd
from datetime import datetime as DT
from datetime import timedelta
from copy import deepcopy as DC
import matplotlib.pylab as plt
import sys

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute

try:
    import pandas_datareader.data as web
except ImportError:
    print("You need to install the pandas_datareader. Run pip install pandas_datareader.")

from tsfresh import extract_features
# from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, EfficientFCParameters
from CustomizeFCParameters import CustomizeFCParameters

sMyPath = r'C:\UCSD_ML_Capstone'
os.chdir(sMyPath)

##### Feature Generation #####
### Return the file names
# from os import listdir
# from os.path import isfile, join

datapath = sMyPath + r'\01_Input\01_03_ProcessedData\Tiingo_Stock_daily_Outlier_Removed'
savepath = sMyPath + r'\01_Input\01_03_ProcessedData\Tiingo_Stock_daily_Transformed'
# AvaTickers_df = pd.read_csv(sMyPath + r"\01_Input\01_01_DataCodes\Stock_Techonology_Semiconductors.csv")
# AvaTickers  = AvaTickers_df..values
AvaTickers_df = pd.read_csv(sMyPath + r"\01_Input\01_01_DataCodes\sNA_Select_Tickers.csv")
AvaTickers = AvaTickers_df[(AvaTickers_df.Sector=='Technology')&(AvaTickers_df.Industry=='Semiconductors')].Tickers.values


### Data transformation 1 ###
use_data_of_list = np.array([1,2,3,6])*21
S_time = DT.now()
AvgSeries = pd.Series(np.empty(len(AvaTickers)),name = 'AvgSeries')
EstEndSeries = pd.Series(np.empty(len(AvaTickers)),name = 'EstEndSeries')

## add module to path
Adding_Path = r'C:\UCSD_ML_Capstone\02_Code\02_01_Data_Process'
if(Adding_Path not in sys.path):
    sys.path.append(Adding_Path)    
from CustomizeFCParameters import CustomizeFCParameters, Exponential_Mean, Lag_DF

TickerStart = 0
iTicker = TickerStart
for iTicker in np.arange(TickerStart, len(AvaTickers)):
    Loop_start = DT.now()
    Ticker = AvaTickers[iTicker]

    # Ticker = 'LSCC'
    print(r"Running " + Ticker + r" ("+str(iTicker+1)+r"/"+str(len(AvaTickers))+")")
    Sdata = pd.read_pickle(datapath + r"\\" + Ticker + "_daily.zip") 

    Sdata = Sdata[['date', 'volume', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'divCash', 'splitFactor']]
    
    ## log ##
    Sdata[['volume_log', 'adjOpen_log', 'adjHigh_log', 'adjLow_log', 'adjClose_log']] = np.log(Sdata[['volume', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose']])
    
    ## spread ##
    Sdata['HL_Spread'] = Sdata['adjHigh'] - Sdata['adjLow'] 
    Sdata['OC_Spread'] = Sdata['adjClose'] - Sdata['adjOpen']
    Sdata['HL_Log_Spread'] = Sdata['adjHigh_log'] - Sdata['adjLow_log'] 
    Sdata['OS_Log_Spread'] = Sdata['adjClose_log'] - Sdata['adjOpen_log']
    
    ## log return and volatility (ret square) ##
    Sdata['Ret_D'] = Sdata['adjClose_log'].diff()
    Sdata['Vol_D'] = Sdata['Ret_D']**2
    Sdata['Ret_W'] = Sdata['Ret_D'].rolling(5).sum()
    Sdata['Vol_W'] = Sdata['Ret_W']**2
    Sdata['Ret_M'] = Sdata['Ret_D'].rolling(21).sum()
    Sdata['Vol_M'] = Sdata['Ret_M']**2
    
    ## Interpolate NAs ##
    Sdata.iloc[:,1:] = Sdata.iloc[:,1:].interpolate(method='linear', limit_direction='forward', axis=0)
    # Sdata = Sdata.dropna()
    Sdata['Ticker'] = Ticker
    
    ### tsfresh generation ###
    df = Sdata[['Ticker', 'date', 'volume_log', 'adjClose_log','HL_Log_Spread', 'OS_Log_Spread',\
                'Ret_D','Vol_D', 'Ret_W', 'Ret_M', 'Vol_W', 'Vol_M']].copy()
    df['date'] = [pd.to_datetime(x) for x in df['date']]

    # used for later (customized features)
    df_raw = df.copy()  
 
    df.dropna(inplace=True)
    del Sdata
    
    ## Extend the data
    from tsfresh.utilities.dataframe_functions import roll_time_series
    melted = df.melt(id_vars=['Ticker','date'])
    # melted.columns
    
    X = []
    for use_data_of in use_data_of_list:
        
        rolled = roll_time_series(melted.copy(), 
                                  column_id = 'Ticker', 
                                  column_sort = 'date',
                                  column_kind='variable', 
                                  rolling_direction=1,
                                  max_timeshift=use_data_of-1,
                                  min_timeshift=use_data_of-1).reset_index(drop=True)
        if(use_data_of == 63):
            rolled_cus = rolled.copy()
        ## extract the features
        X_tmp = extract_features(rolled, column_id = 'id', column_sort='date',
                              column_kind='variable', column_value='value',
                              default_fc_parameters = CustomizeFCParameters(), n_jobs=4)
        
        X_tmp.columns = [x+r'_rolls'+str(use_data_of) for x in X_tmp.columns]
        
        if(len(X)<=0):
            X = X_tmp
        else:
            X = pd.concat([X, X_tmp], axis=1)
        
    del melted , X_tmp

    ## remove features with too much NAs
    AllNa_Cols1 = X.apply(lambda x: sum((x.isna()) | (np.isinf(abs(x)))) >= X.shape[0]*0.45, axis=0)
    AllNa_Cols2 = X.iloc[(max(use_data_of_list)*2):,:].apply(lambda x: sum((x.isna()) | (np.isinf(abs(x)))) >= X.shape[0]*0.01, axis=0)
    AllNa_Cols = (AllNa_Cols1 | AllNa_Cols2)
    X = X.loc[:,~AllNa_Cols.values]

    ## Some Customized Features
    # Exponential mean and vol
    ExpVar = rolled_cus.groupby(['id','variable']).\
        apply(lambda x: Exponential_Mean(x.value, length = len(x.value), decay = 0.2**(1/21)))
    ExpVar = pd.DataFrame(ExpVar)
    ExpVar['Date'] = ExpVar.index.map(lambda x: x[0][1]).values
    ExpVar['Feature'] = ExpVar.index.map(lambda x: x[1]).values
    # ExpVar.reset_index(drop=True, inplace=True)
    ExpVar.columns = ['Value', 'Date', 'Feature']
    ExpVar = ExpVar.pivot_table(values = 'Value', index= ['Date'], columns=['Feature'])
    ExpVar.columns = [x + r'_EM' for x in ExpVar.columns]

    # Lag terms daily
    df_D = df_raw[['volume_log', 'adjClose_log','HL_Log_Spread', 'OS_Log_Spread', 'Ret_D','Vol_D']].copy()
    df_D = Lag_DF(df_D, laglist = list(range(1,6)) + ([10,15,21,63]))
    df_D.index = df_raw.date

    # Lag terms weekly
    df_W = df_raw[['Ret_W','Vol_W']].copy()
    df_W = Lag_DF(df_W, laglist = [1,2,3,4,8,12])
    df_W.index = df_raw.date
    
    # Lag terms monthly
    df_M = df_raw[['Ret_M','Vol_M']].copy()
    df_M = Lag_DF(df_M, laglist = [1,2,3])
    df_M.index = df_raw.date

    ## Merge tsfresh and customized
    X.index = X.index.map(lambda x: x[1]).values
    X = X.merge(ExpVar, how = 'outer', left_index=True, right_index=True)
    X = X.merge(df_D, how = 'outer', left_index=True, right_index=True)
    X = X.merge(df_W, how = 'outer', left_index=True, right_index=True)
    X = X.merge(df_M, how = 'outer', left_index=True, right_index=True)

    ## Remove NAs
    X = X.dropna()

    ## Save
    X.to_pickle(savepath + r"\\" + Ticker + r"_tsfresh.zip")
    del X, df_raw, df_D, df_W, df_M
    
    ## Estimate finish time
    Loop_end = DT.now()
    LoopTime = Loop_end-Loop_start
    TotalTime = Loop_end-S_time
    AvgTime = (TotalTime)/(iTicker -TickerStart +1)
    EstEnd = DT.now() + (len(AvaTickers) - iTicker - TickerStart - 1)*AvgTime
    
    AvgSeries[iTicker] = AvgTime
    EstEndSeries[iTicker] = EstEnd
    print('Done No.'+ str(iTicker+1) + " " +Ticker+ ";\n" +
          "Loop time:" + str(LoopTime) + "\n" +
          "Total time:" + str(TotalTime) + "\n" +
          "Average time:" + str(AvgTime) + "\n" +
          "Estimate end:" + str(EstEnd) + '\n'  )

