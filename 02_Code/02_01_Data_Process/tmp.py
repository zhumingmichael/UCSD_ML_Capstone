%reset -f

import os
import numpy as np
import pandas as pd
from datetime import datetime as DT
from datetime import timedelta
from copy import deepcopy as DC

sMyPath = r'C:\UCSD_ML_Capstone'
os.chdir(sMyPath)



##### Data Analysis #####
### Return the file names
from os import listdir
from os.path import isfile, join

datapath = sMyPath + r'\01_Input\01_02_RawData\Tiingo_Stock_daily'
savepath = sMyPath + r'\01_Input\01_03_ProcessedData\Tiingo_Stock_daily_Outlier_Removed'

# Get the file names of the data
filesnames = listdir(datapath)
# Extract the tickers from the file names
AvaTickers = [x.split("_daily.csv")[0] for x in filesnames]
AvaTickers = np.sort(AvaTickers)

for iTicker in range(len(AvaTickers)):
    ## Import the stock data
    Ticker = AvaTickers[iTicker]
    
    Sdata = pd.read_csv(datapath + r"\\" + Ticker + "_daily.csv") 
    Sdata['date'] = [DT.strptime(x[:10], "%Y-%m-%d") for x in Sdata['date']]
    Sdata = Sdata.loc[:,['date', 'adjOpen', 'adjHigh','adjLow', 'adjClose', 'adjVolume']]
    
    
    Sdata.loc[:,['adjOpen', 'adjHigh','adjLow', 'adjClose']] =\
        Sdata.loc[:,['adjOpen', 'adjHigh','adjLow', 'adjClose']].\
            apply(lambda x: Outlier_Detection(x,returnType = 'relative',\
                                              Chebyshev_k=20, method = 'Chebyshev')[0])

    Sdata.loc[:,'adjVolume'] =  Outlier_Detection(Sdata.loc[:,'adjVolume'],\
                                                  returnType = 'absolute',\
                                                  Chebyshev_k=20, \
                                                  method = 'Chebyshev')[0]

