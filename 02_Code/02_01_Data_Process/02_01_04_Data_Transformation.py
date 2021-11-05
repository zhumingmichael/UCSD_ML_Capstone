%reset -f

import os
import numpy as np
import pandas as pd
from datetime import datetime as DT
from datetime import timedelta
from copy import deepcopy as DC
import matplotlib.pylab as plt

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute

try:
    import pandas_datareader.data as web
except ImportError:
    print("You need to install the pandas_datareader. Run pip install pandas_datareader.")

# from sklearn.ensemble import AdaBoostRegressor

sMyPath = r'C:\UCSD_ML_Capstone'
os.chdir(sMyPath)

##### Feature Generation #####
### Return the file names
from os import listdir
from os.path import isfile, join

datapath = sMyPath + r'\01_Input\01_03_ProcessedData\Tiingo_Stock_daily_Outlier_Removed'
savepath = sMyPath + r'\01_Input\01_03_ProcessedData\Tiingo_Stock_daily_Transformed'
AvaTickers_df = pd.read_csv(sMyPath + r"\01_Input\01_01_DataCodes\Stock_Techonology_Semiconductors.csv")
AvaTickers  = AvaTickers_df.Tickers.values


### Data transformation 1 ###
def Cross_Transform():
    use_data_of = 21*3
    S_time = DT.now()
    AvgSeries = pd.Series(np.empty(len(AvaTickers)),name = 'AvgSeries')
    EstEndSeries = pd.Series(np.empty(len(AvaTickers)),name = 'EstEndSeries')
    
    for iTicker in np.arange(len(AvaTickers)):
        Loop_start = DT.now()
        Ticker = AvaTickers[iTicker]
    
        # Ticker = 'LSCC'
        print(r"Running " + Ticker + r" ("+str(iTicker+1)+r"/"+str(len(AvaTickers))+")")
        Sdata = pd.read_csv(datapath + r"\\" + Ticker + "_daily.csv") 
    
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
        Sdata = Sdata.interpolate(method='linear', limit_direction='forward', axis=0)
        # Sdata = Sdata.dropna()
        Sdata['Ticker'] = Ticker
        
        ### tsfresh generation ###
        df = Sdata[['Ticker', 'date', 'volume_log', 'adjClose_log','HL_Log_Spread', 'OS_Log_Spread',\
                    'Ret_D','Vol_D', 'Ret_W', 'Ret_M', 'Vol_W', 'Vol_M']].copy()
        
        df['date'] = [pd.to_datetime(x) for x in df['date']]
        
        df.dropna(inplace=True)
        
        del Sdata
        
        from tsfresh.utilities.dataframe_functions import roll_time_series
        melted = df.melt(id_vars=['Ticker','date'])
        melted.columns
        rolled = roll_time_series(melted.copy(), 
                                  column_id = 'Ticker', 
                                  column_sort = 'date',
                                  column_kind='variable', 
                                  rolling_direction=1,
                                  max_timeshift=use_data_of-1,
                                  min_timeshift=use_data_of-1).reset_index(drop=True)
        
        del melted
        
        
        from tsfresh import extract_features
        from tsfresh.feature_extraction.settings import EfficientFCParameters
        
        X = extract_features(rolled, column_id = 'id', column_sort='date',
                              column_kind='variable', column_value='value',
                              default_fc_parameters = EfficientFCParameters(), n_jobs=4)
        
        ## Remove NAs
        AllNa_Cols = X.apply(lambda x: sum(x.isna()) >= X.shape[0]*0.01, axis=0)
        X = X.loc[:,~AllNa_Cols.values].copy()
        X = X.dropna()    
        
        ## Save
        X.to_pickle(savepath + r"\\" + Ticker + r"_tsfresh.zip")
    
        del X
        ## Estimate finish time
        Loop_end = DT.now()
        LoopTime = Loop_end-Loop_start
        TotalTime = Loop_end-S_time
        AvgTime = (TotalTime)/(iTicker+1)
        EstEnd = DT.now() + (len(AvaTickers) - iTicker - 1)*AvgTime
        
        AvgSeries[iTicker] = AvgTime
        EstEndSeries[iTicker] = EstEnd
        print('Done No.'+ str(iTicker+1) + " " +Ticker+ ";\n" +
              "Loop time:" + str(LoopTime) + "\n" +
              "Total time:" + str(TotalTime) + "\n" +
              "Average time:" + str(AvgTime) + "\n" +
              "Estimate end:" + str(EstEnd) + '\n'  )
    
    

### Data transformation 2 ###
S_time = DT.now()
AvgSeries = pd.Series(np.empty(len(AvaTickers)),name = 'AvgSeries')
EstEndSeries = pd.Series(np.empty(len(AvaTickers)),name = 'EstEndSeries')

Price_DF = []
for iTicker in np.arange(len(AvaTickers)):
    Loop_start = DT.now()
    Ticker = AvaTickers[iTicker]

    # Ticker = 'LSCC'
    print(r"Running " + Ticker + r" ("+str(iTicker+1)+r"/"+str(len(AvaTickers))+")")
    Sdata = pd.read_csv(datapath + r"\\" + Ticker + "_daily.csv") 

    Sdata['date'] = [pd.to_datetime(x) for x in Sdata['date']]
    Price_Series = Sdata.adjClose
    Price_Series.index = Sdata['date']
    Price_Series.name = Ticker
    
    if len(Price_DF) <= 0:
        Price_DF = pd.DataFrame(Price_Series)
    else:
        Price_DF = Price_DF.merge(Price_Series, how='outer',left_index=True, right_index=True)
    
    ## Estimate finish time
    Loop_end = DT.now()
    LoopTime = Loop_end-Loop_start
    TotalTime = Loop_end-S_time
    AvgTime = (TotalTime)/(iTicker+1)
    EstEnd = DT.now() + (len(AvaTickers) - iTicker - 1)*AvgTime
    
    AvgSeries[iTicker] = AvgTime
    EstEndSeries[iTicker] = EstEnd
    print('Done No.'+ str(iTicker+1) + " " +Ticker+ ";\n" +
          "Loop time:" + str(LoopTime) + "\n" +
          "Total time:" + str(TotalTime) + "\n" +
          "Average time:" + str(AvgTime) + "\n" +
          "Estimate end:" + str(EstEnd) + '\n'  )
    

## Transformation ##
Price_log = np.log(Price_DF).copy()
# Price_D = np.log(Price_DF).copy()
Price_log.columns = [str(x)+r"_log" for x in Price_DF.columns]

Ret_D = Price_log.diff()
Ret_D.columns = [str(x)+r"_Ret_D" for x in Price_DF.columns]

Vol_D = Ret_D**2
Vol_D.columns = [str(x)+r"_Vol_D" for x in Price_DF.columns]

Ret_W = Ret_D.rolling(5).sum()
Ret_W.columns = [str(x)+r"_Ret_W" for x in Price_DF.columns]

Vol_W = Ret_W**2
Vol_W.columns = [str(x)+r"_Vol_W" for x in Price_DF.columns]

Ret_M = Ret_D.rolling(21).sum()
Ret_M.columns = [str(x)+r"_Ret_M" for x in Price_DF.columns]

Vol_M = Ret_M**2
Vol_M.columns = [str(x)+r"_Vol_M" for x in Price_DF.columns]


Ret_Q = Ret_D.rolling(63).sum()
Ret_Q.columns = [str(x)+r"_Ret_Q" for x in Price_DF.columns]

Vol_Q = Ret_Q**2
Vol_Q.columns = [str(x)+r"_Vol_Q" for x in Price_DF.columns]


Price = pd.concat([Price_log, Ret_D, Vol_D,\
                   Ret_W, Vol_W, Ret_M, Vol_M, Ret_Q, Vol_Q], axis=1)

Price.to_pickle(savepath + r"\Price1.zip")

# Price_D = pd.concat([Price_D, Vol_D], axis=1)
Price_D = Ret_D

### tsfresh generation ###
df = Price_D.copy()
df['date'] =  df.index
df.dropna(inplace=True)

from tsfresh.utilities.dataframe_functions import roll_time_series
use_data_of = 21*3
melted = df.melt(id_vars=['date'])
melted.columns
melted['id']=1
rolled = roll_time_series(melted.copy(), 
                          column_id = 'id', 
                          column_sort = 'date',
                          column_kind='variable', 
                          rolling_direction=1,
                          max_timeshift=use_data_of-1,
                          min_timeshift=use_data_of-1).reset_index(drop=True)

del melted


from tsfresh import extract_features
from tsfresh.feature_extraction.settings import EfficientFCParameters

X = extract_features(rolled, column_id = 'id', column_sort='date',
                      column_kind='variable', column_value='value',
                      default_fc_parameters = EfficientFCParameters(), n_jobs=4)

## Remove NAs
AllNa_Cols = X.apply(lambda x: sum(x.isna()) >= X.shape[0]*0.01, axis=0)
X = X.loc[:,~AllNa_Cols.values].copy()
X = X.dropna()    

## Save
X.to_pickle(savepath + r"\Ret_D_tsfresh.zip")

del X
    
    
a = pd.Series(r"Running " + Ticker + r" ("+str(iTicker+1)+r"/"+str(len(AvaTickers))+")")
a.to_csv(savepath + r"\\" + 'a.csv')
    