%reset -f

import os
import numpy as np
import pandas as pd
from datetime import datetime as DT
from datetime import timedelta
from copy import deepcopy as DC
from sklearn.neighbors import LocalOutlierFactor as LOF
from itertools import chain

sMyPath = r'C:\UCSD_ML_Capstone'
os.chdir(sMyPath)

##### Functions #####
# Se = DC(Sdata.adjClose)
# Se = Sdata.iloc[:,3]
def Outlier_Detection(Se, returnType = 'relative',Chebyshev_k=20,\
                      method = 'Chebyshev',n_neighbors = 20):
    
    lSe = DC(Se)
    if returnType == 'relative':
        rSe = np.log(lSe).diff()
    elif returnType == 'absolute':
        rSe = lSe.diff()
    else:
        raise ValueError('returnType should be absolute or relative.')
           
        
    if method == 'Chebyshev':
        ## calculate the deviation
        Dev = np.abs(rSe - rSe.mean())/rSe.std()
        ## remove the outliers
        remove_index = lSe[Dev>=Chebyshev_k].index 
    elif method == 'LOF':
        ## Local Outlier Factor model
        clf = LOF(n_neighbors=n_neighbors, n_jobs=4)
        ## Detect the outliers
        ifOutlier = clf.fit_predict(rSe.dropna().values.reshape(-1,1))
        ifOutlier = pd.Series(ifOutlier, index = range(1,len(ifOutlier)+1))
        remove_index = ifOutlier[ifOutlier==-1].index
        
    lSe[remove_index] = np.nan

    return([lSe, remove_index])    
    # Series, Removal = Outlier_Detection(Sdata.adjClose, 8, 'relative')


##### Outlier Remove #####
datapath = sMyPath + r'\01_Input\01_02_RawData\Tiingo_Stock_daily'
savepath = sMyPath + r'\01_Input\01_03_ProcessedData\Tiingo_Stock_daily_Outlier_Removed'
AvaTickers_df = pd.read_excel(sMyPath + r"\01_Input\01_01_DataCodes\sNA_Select_Tickers.xlsx")
AvaTickers  = AvaTickers_df.Tickers.values


### Set the threshold of low volume
from scipy.stats import norm
P_sVol = 5/1000
Pert_sVol = norm.ppf(P_sVol)

TickerSummary=[]
OutliersSum = []

AvgSeries = pd.Series(np.empty(len(AvaTickers)),name = 'AvgSeries')
EstEndSeries = pd.Series(np.empty(len(AvaTickers)),name = 'EstEndSeries')
## Time the loop
S_time = DT.now()
for iTicker in range(len(AvaTickers)):
    Loop_start = DT.now()

    ## Import the stock data
    Ticker = AvaTickers[iTicker]
    
    Sdata = pd.read_csv(datapath + r"\\" + Ticker + "_daily.csv") 
    Sdata['date'] = [DT.strptime(x[:10], "%Y-%m-%d") for x in Sdata['date']]
    
    ## relatives
    RelativeCols = Sdata[['open', 'high', 'low', 'close', 'adjOpen', 'adjHigh',\
                          'adjLow', 'adjClose']]
    RelativeCols = RelativeCols.apply(lambda x: Outlier_Detection(x,returnType = 'relative',\
                                              Chebyshev_k=20, method = 'Chebyshev')[0])
        
    Sdata[['open', 'high', 'low', 'close', 'adjOpen', 'adjHigh',\
                          'adjLow', 'adjClose']] = RelativeCols
        
    ## absolute
    AbsoluteCols = Sdata[['volume', 'adjVolume']]
    AbsoluteCols = AbsoluteCols.apply(lambda x: Outlier_Detection(x,returnType = 'absolute',\
                                              Chebyshev_k=20, method = 'Chebyshev')[0])
        
    Sdata[['volume', 'adjVolume']] = AbsoluteCols
    Sdata = Sdata
    
    Sdata = Sdata[Sdata['date']>=DT(2010,1,1)]

    ## Save processed data
    Sdata.to_csv(savepath + r"\\" + Ticker + "_daily.csv", index=False)
    ## Check Na amount
    OutlierStats = Sdata.apply(lambda x: sum(np.isnan(x)) )
    OutlierStats.name = Ticker

    if len(OutliersSum)<=0:
        OutliersSum = OutlierStats
    else:
        OutliersSum = pd.concat([OutliersSum,OutlierStats],axis=1)
        
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



AvgSeries.plot()
EstEndSeries.plot()


# ## Data path
# datapath = sMyPath + r'\01_Input\01_2_RawData\Tiingo_Stock_daily'

# Ticker = 'WY'
# Sdata = pd.read_excel(datapath + r"\\" + Ticker + "_daily.xlsx")
# Sdata.columns

# Sdata['date'] = [DT.strptime(x[:10], "%Y-%m-%d") for x in Sdata['date']]
# Sdata = Sdata.loc[:,['date', 'adjOpen', 'adjHigh','adjLow', 'adjClose', 'adjVolume']]
# Sdata.iloc[:,1:] = Sdata.iloc[:,1:].apply(lambda x: Outlier_Detection(x, 8, 'relative')[0])




# x = Sdata.iloc[:,1]
# sum(np.isnan(x))


