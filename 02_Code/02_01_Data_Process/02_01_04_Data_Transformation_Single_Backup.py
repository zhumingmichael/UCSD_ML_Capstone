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
use_data_of = 21*3
S_time = DT.now()
AvgSeries = pd.Series(np.empty(len(AvaTickers)),name = 'AvgSeries')
EstEndSeries = pd.Series(np.empty(len(AvaTickers)),name = 'EstEndSeries')

## add module to path
Adding_Path = r'C:\UCSD_ML_Capstone\02_Code\02_01_Data_Process'

if(Adding_Path not in sys.path):
    sys.path.append(Adding_Path)    
from CustomizeFCParameters import CustomizeFCParameters
# CustomizeFCParameters()

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
    
    
    from tsfresh import extract_features,extract_relevant_features
    from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, EfficientFCParameters
    
    # from tsfresh import extract_relevant_features
    # from tsfresh.feature_extraction import ComprehensiveFCParameters
    
    # aa = ComprehensiveFCParameters()
    # bb = EfficientFCParameters()
    
    # file2 = open("./ComprehensiveFCParameters.txt","w+")
    # file2.write(str(aa))
    # file2.close()
    
    
    # import pandas as pd
    # from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
    # from tsfresh import extract_features
    # download_robot_execution_failures()
    # timeseries, y = load_robot_execution_failures()
    # extracted_features = extract_features(timeseries, column_id="id",
    #                                       column_sort="time", default_fc_parameters = CustomizeFCParameters())
    # a_ = pd.Series(extracted_features.columns)
    # a_.to_csv('a_.csv')
    
    
    
    # X = extract_features(rolled, column_id = 'id', column_sort='date',
    #                       column_kind='variable', column_value='value',
    #                       default_fc_parameters = EfficientFCParameters(), n_jobs=4)
    
    X2 = extract_features(rolled, column_id = 'id', column_sort='date',
                          column_kind='variable', column_value='value',
                          default_fc_parameters = CustomizeFCParameters(), n_jobs=4)
    
    CustomizeFCParameters()
    b_ = pd.Series(X2.columns)
    b_.to_csv('CustomizeFCParameters.csv')    
    
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

