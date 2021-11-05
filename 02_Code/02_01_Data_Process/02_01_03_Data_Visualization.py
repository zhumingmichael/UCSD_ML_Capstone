%reset -f

import os
import numpy as np
import pandas as pd
from datetime import datetime as DT
from datetime import timedelta
from copy import deepcopy as DC

sMyPath = r'C:\UCSD_ML_Capstone'
os.chdir(sMyPath)

##### 1. Stock Select based on cap #####
### Stock Screener: industry and sector
Stock_Screener_Nas = pd.read_csv(r'C:\UCSD_ML_Capstone\01_Input\01_01_DataCodes\Stock_Sector_Industry.csv')
Stock_Screener_Nas.columns
## Tiingo Available tickers
AvaTickers = pd.read_csv(r'C:\UCSD_ML_Capstone\01_Input\01_01_DataCodes\Tiingo_Available_Tickers.csv')

## Remove nan and 0 cap
Stock_Screener = Stock_Screener_Nas[['Symbol', 'Market Cap','Volume', 'Sector', 'Industry']]
Stock_Screener.columns = ['Tickers', 'Market Cap','Volume', 'Sector', 'Industry']
Stock_Screener.loc[Stock_Screener['Market Cap']<=0,'Market Cap'] = np.nan
Stock_Screener = Stock_Screener.dropna()

## merge
Stock_Screener_Ava_Ori = Stock_Screener.merge(AvaTickers, how = 'inner', left_on='Tickers', right_on='Tickers')

### Calculate the Capitals
Market_Cap = Stock_Screener_Nas['Market Cap'].sum()
Industry_count = Stock_Screener_Nas[['Symbol', 'Industry']].groupby(by = 'Industry').count()
Sector_count = Stock_Screener_Nas[['Symbol', 'Sector']].groupby(by = 'Sector').count()
Industry_Cap = Stock_Screener_Nas[['Market Cap', 'Industry']].groupby(by = 'Industry').sum()
Sector_Cap = Stock_Screener_Nas[['Market Cap', 'Sector']].groupby(by = 'Sector').sum()

## Thresholds
Th_Quan_Market = 0.8
Th_Quan_Industry = 0.5
Th_Quan_Sector = 0.5

Market_Cap_Thres = Stock_Screener_Nas['Market Cap'].quantile(Th_Quan_Market)

Industry_Cap_Thres = Stock_Screener_Nas[['Market Cap', 'Industry']].\
    groupby(by = 'Industry').quantile(Th_Quan_Industry)
Industry_Cap_Thres.columns = ['Industry_Cap_Thres']


Sector_Cap_Thres = Stock_Screener_Nas[['Market Cap', 'Sector']].\
    groupby(by = 'Sector').quantile(Th_Quan_Sector)
Sector_Cap_Thres.columns = ['Sector_Cap_Thres']

Stock_Screener_Ava = DC(Stock_Screener_Ava_Ori)

Stock_Screener_Ava['Market_Cap_Thres'] = Market_Cap_Thres
Stock_Screener_Ava = Stock_Screener_Ava.merge(Industry_Cap_Thres,\
                                              left_on='Industry', right_index=True)
Stock_Screener_Ava = Stock_Screener_Ava.merge(Sector_Cap_Thres,\
                                              left_on='Sector', right_index=True)

Stock_Screener_Ava.columns

Stock_Screener_Ava['Market_Cap_Pass'] = Stock_Screener_Ava['Market Cap']>=\
    Stock_Screener_Ava['Market_Cap_Thres']
Stock_Screener_Ava['Industry_Cap_Pass'] = Stock_Screener_Ava['Market Cap']>=\
    Stock_Screener_Ava['Industry_Cap_Thres']
Stock_Screener_Ava['Sector_Cap_Pass'] = Stock_Screener_Ava['Market Cap']>=\
    Stock_Screener_Ava['Sector_Cap_Thres']

Picked = Stock_Screener_Ava['Market_Cap_Pass'] |\
            (Stock_Screener_Ava['Industry_Cap_Pass'] | \
            Stock_Screener_Ava['Sector_Cap_Pass'])

sum(Picked)
Cap_Pass_Tickers = Stock_Screener_Ava.loc[Picked,'Tickers'].values


##### 2. Stock Select based on volume 0 #####
datapath = sMyPath + r'\01_Input\01_02_RawData\Tiingo_Stock_daily'
from scipy.stats import norm
P_sVol = 5/1000
Pert_sVol = norm.ppf(P_sVol)

TickerSummary=[]
AvgSeries = pd.Series(np.empty(len(Cap_Pass_Tickers)),name = 'AvgSeries')
EstEndSeries = pd.Series(np.empty(len(Cap_Pass_Tickers)),name = 'EstEndSeries')
ZeroLenTicker = []
## Time the loop
S_time = DT.now()
for iTicker in range(len(Cap_Pass_Tickers)):
    Loop_start = DT.now()   

    ## Import the stock data
    Ticker = Cap_Pass_Tickers[iTicker]
    
    Sdata = pd.read_csv(datapath + r"\\" + Ticker + "_daily.csv")
    # Sdata.columns
    if len(Sdata) >0:
        
        StartDate = pd.Series(Sdata.date[0], ['Start_date'])
        VolDes = Sdata.volume.describe()
        
        ## Calculate the thresholds of small volumes
        Vol_log = np.log(Sdata.volume[Sdata.volume>0])
        Thres_sVol = np.exp(Vol_log.mean() + Pert_sVol*Vol_log.std())
        ## The proportion of small volumes
        sVol_len = pd.Series(sum(Sdata.volume <= Thres_sVol), index = ['sVol_len'])
        sVol_prop = pd.Series(sVol_len[0]/VolDes['count'], index = ['sVol_prop'])
        ## The proportion of 0 volumnes
        Vol0_len = pd.Series(sum(Sdata.volume <= 0), index = ['Vol0_len'])
        Vol0_prop = pd.Series(Vol0_len[0]/VolDes['count'], index = ['Vol0_prop'])
        
        ## Output
        TickerVol = pd.concat([StartDate, VolDes,sVol_len,sVol_prop,Vol0_len,Vol0_prop])
        TickerVol.name = Ticker
        
        if len(TickerSummary)<=0:
            TickerSummary = TickerVol
        else:
            TickerSummary = pd.concat([TickerSummary,TickerVol],axis=1)
        
    else:
        ZeroLenTicker.append(Ticker)
        
    ## Estimate finish time
    Loop_end = DT.now()
    LoopTime = Loop_end-Loop_start
    TotalTime = Loop_end-S_time
    AvgTime = (TotalTime)/(iTicker+1)
    EstEnd = DT.now() + (len(Cap_Pass_Tickers) - iTicker - 1)*AvgTime
    
    AvgSeries[iTicker] = AvgTime
    EstEndSeries[iTicker] = EstEnd
    print('Done No.'+ str(iTicker+1) + " " +Ticker+ ";\n" +
          "Loop time:" + str(LoopTime) + "\n" +
          "Total time:" + str(TotalTime) + "\n" +
          "Average time:" + str(AvgTime) + "\n" +
          "Estimate end:" + str(EstEnd) + '\n'  )



TickerSummary = TickerSummary.T

Ana = DC(TickerSummary)
Ana.columns
## Select the stock IPO before 2010-01-01
Ana = Ana[[DT.strptime(x[:10], "%Y-%m-%d")<= DT(2010,1,1) for x in Ana['Start_date']]]
Ana = Ana[Ana.sVol_prop<=2.5/100]

sVol_Select_Tickers = pd.Series(list(Ana.index))
sVol_Select_Tickers.to_excel(sMyPath + r"\01_Input\sVol_Select_Tickers.xlsx",index=False)




##### 3. Data Analysis #####
## 
P_l_Adj = []
P_l_Ori = []
V_l_Adj = []
V_l_Ori = []

## Time the loop
S_time = DT.now()
for iTicker in range(len(sVol_Select_Tickers)):
    Loop_start = DT.now()   

    ## Import the stock data
    Ticker = sVol_Select_Tickers[iTicker]
    
    Sdata = pd.read_csv(datapath + r"\\" + Ticker + "_daily.csv") 
    Sdata['date'] = [DT.strptime(x[:10], "%Y-%m-%d") for x in Sdata['date']]
    Sdata = Sdata.drop_duplicates(subset = ['date'],keep ='last')
    
    ## Dependent variables
    P_l_A = pd.Series(Sdata.adjClose.values, index = Sdata.date)
    P_l_O = pd.Series(Sdata.close.values, index = Sdata.date)
    V_l_A = pd.Series(Sdata.adjVolume.values, index = Sdata.date)
    V_l_O = pd.Series(Sdata.volume.values, index = Sdata.date)

    P_l_A.name = Ticker
    P_l_O.name = Ticker
    V_l_A.name = Ticker
    V_l_O.name = Ticker

    if len(P_l_Adj)<=0:
        P_l_Adj = P_l_A
        P_l_Ori = P_l_O
        V_l_Adj = V_l_A
        V_l_Ori = V_l_O
    else:
        P_l_Adj = pd.concat([P_l_Adj, P_l_A], axis = 1)
        P_l_Ori = pd.concat([P_l_Ori, P_l_O], axis = 1)
        V_l_Adj = pd.concat([V_l_Adj, V_l_A], axis = 1)
        V_l_Ori = pd.concat([V_l_Ori, V_l_O], axis = 1)

    ## Estimate finish time
    Loop_end = DT.now()
    LoopTime = Loop_end-Loop_start
    TotalTime = Loop_end-S_time
    AvgTime = (TotalTime)/(iTicker+1)
    EstEnd = DT.now() + (len(sVol_Select_Tickers) - iTicker - 1)*AvgTime
    
    AvgSeries[iTicker] = AvgTime
    EstEndSeries[iTicker] = EstEnd
    print('Done No.'+ str(iTicker+1) + " " +Ticker+ ";\n" +
          "Loop time:" + str(LoopTime) + "\n" +
          "Total time:" + str(TotalTime) + "\n" +
          "Average time:" + str(AvgTime) + "\n" +
          "Estimate end:" + str(EstEnd) + '\n'  )




##### 4. Time cut and removal based on existing NAs #####
P_l_Adj.to_csv('P_l_Adj.csv')
P_l_Ori.to_csv('P_l_Ori.csv')
V_l_Adj.to_csv('V_l_Adj.csv')
V_l_Ori.to_csv('V_l_Ori.csv')

P_l_Adj2010 = DC(P_l_Adj[P_l_Adj.index>=DT(2010,1,1)])
# ## Time the loop
# S_time = DT.now()
# for iTicker in range(len(sVol_Select_Tickers)):
#     Loop_start = DT.now()   

#     Ticker = sVol_Select_Tickers[iTicker]

#     plt_title = Ticker + "_AdjClose"
#     savepath = r'C:\UCSD_ML_Capstone\01_Input\01_02_RawData\Daily_Plot\Adjust_Close'
#     pp = P_l_Adj2010[[Ticker]].plot(title=plt_title, figsize=(15,12))
#     pp.figure.savefig(savepath+r'\\'+Ticker+r'_daily.png')

#     ## Estimate finish time
#     Loop_end = DT.now()
#     LoopTime = Loop_end-Loop_start
#     TotalTime = Loop_end-S_time
#     AvgTime = (TotalTime)/(iTicker+1)
#     EstEnd = DT.now() + (len(sVol_Select_Tickers) - iTicker - 1)*AvgTime
    
#     AvgSeries[iTicker] = AvgTime
#     EstEndSeries[iTicker] = EstEnd
#     print('Done No.'+ str(iTicker+1) + " " +Ticker+ ";\n" +
#           "Loop time:" + str(LoopTime) + "\n" +
#           "Total time:" + str(TotalTime) + "\n" +
#           "Average time:" + str(AvgTime) + "\n" +
#           "Estimate end:" + str(EstEnd) + '\n'  )


Na_stats = P_l_Adj2010.apply(lambda x: sum(x.isna()))
sNA_Select_Tickers = pd.Series(np.setdiff1d(sVol_Select_Tickers.values,\
                                             np.array(['VAL','ENR']) ))

    
sNA_Select_Tickers.to_excel(sMyPath + r"\01_Input\sNA_Select_Tickers.xlsx",index=False)
