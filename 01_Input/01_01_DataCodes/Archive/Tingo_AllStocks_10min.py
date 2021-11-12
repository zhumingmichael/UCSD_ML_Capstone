%reset -f
import requests
import pandas as pd
import numpy as np
from datetime import datetime as DT
from datetime import timedelta
import os
from time import sleep
from itertools import chain

# def main():
os.chdir(r'C:\Data')
## Stock Ticker
# StockAll = pd.read_excel('All_Availables.xlsx')
StockAll = pd.read_excel('AllTickers_20211018.xlsx')
# StockAll = pd.read_excel('SPTicker_20211018.xlsx')

StockTickers  = list(chain(*StockAll.values))

# StockNames = StockAll['Company_Name']
# SPAll = pd.read_excel('SP_Symbol.xlsx')
# SPTickers  = SPAll['Symbol']
# SPNames = SPAll['Company_Name']

## Set token
# Token0 = 'e508d7189bdcf5f2126742277cec9a7dd3451f46'     # famous553299031@gmail.com david7795
# Token1 = 'a4912a52bd8718adae4d33b1dba79bdcc6bb5480'     # zhuming.michaelzm@gmail.com Michaelzm
Token2 = 'ddff0c680755cd0a68288e345d4e422e30e2c07b'     # zhuming.official@gmail.com zhuming

# TokenList = [Token0, Token1, Token2]
TokenList = [Token2]
    
## Request 1-min intraday data
token_num = 0
Token = TokenList[token_num]
TotalTime = 0
ErrorTicker = []
headers = { 'Content-Type': 'application/json' }


##### refresh
S_time = DT.now()
start_iTicker = 0
# start_iTicker = 1406
i_token = 0
wait_itoken = 1400
sleepseconds = 60*10
sleepseconds2 = 0.2

for iTicker in range(start_iTicker,len(StockTickers)):
# for iTicker in range(4583,len(StockTickers)):    
    i_token += 1
    if i_token > wait_itoken:
        print("Out of token at " + str(token_num))
        # reset counters
        i_token = 0
        token_num = 0
        Token = TokenList[token_num]
        # S_time = DT.now()

        # sleep for sometime
        # sleepseconds = 3600-TotalTime.total_seconds()+10

        print('Awake at '+ str(DT.now() + timedelta(seconds=sleepseconds)) )
        sleep(sleepseconds)

    
    
    Ticker = StockTickers[iTicker]
    Loop_start = DT.now()   
    
    try:
        try:
            Current = pd.read_excel("./Tiingo_Stock_10min/" + Ticker + "_10min.xlsx")
    
            LastTime = Current.date[len(Current.date)-1]
            LastTime = LastTime[0:10]
            
            reqdata = requests.get("https://api.tiingo.com/iex/"+Ticker+"/"+\
                                    "prices?startDate="+LastTime+"&"+\
                                    "resampleFreq=10min&"+\
                                    "columns=open,high,low,close,volume&"+\
                                    "token="+Token, headers=headers)
        
            lIntraday = pd.DataFrame(reqdata.json())
            
            NewData = pd.concat([Current, lIntraday], axis = 0)
            NewData = NewData.drop_duplicates(keep ='last')
        
        except FileNotFoundError:
                        
            reqdata = requests.get("https://api.tiingo.com/iex/"+Ticker+"/"+\
                               "prices?startDate=2000-01-01&"+\
                               "resampleFreq=10min&"+\
                               "columns=open,high,low,close,volume&"+\
                               "token="+Token, headers=headers)
            # reqdata = requests.get("https://api.tiingo.com/iex/"+Ticker+"/"+\
            #       "prices?startDate=2021-08-27&"+\
            #       "resampleFreq=1min&"+\
            #       "columns=open,high,low,close,volume&"+\
            #       "token="+Token, headers=headers)
            NewData = pd.DataFrame(reqdata.json())
        
        sleep(sleepseconds2)

        NewData.to_excel("./Tiingo_Stock_10min/" + Ticker + "_10min.xlsx",index=False)

        # calculate time
        Loop_end = DT.now()
        LoopTime = Loop_end-Loop_start
        TotalTime = Loop_end-S_time
        AvgTime = (TotalTime- timedelta(seconds=sleepseconds * ((iTicker-1)//wait_itoken) ) )\
            /(iTicker-start_iTicker+1)
        EstEnd = DT.now() + (len(StockTickers) - iTicker - 1)*AvgTime +\
            timedelta(seconds=sleepseconds * ((len(StockTickers) - iTicker - 1)//wait_itoken) )
        print('Done No.'+ str(iTicker+1) + \
               " " +Ticker+\
              "; Loop time:" + str(LoopTime) +\
              "; Total time:" + str(TotalTime) + \
              "; Average time:" + str(AvgTime) +\
              "; Estimate end:" + str(EstEnd) + '\n'  )


    except ValueError:
        print('Done No.'+ str(iTicker+1) +' '+Ticker + ' is unavailable.')
        ErrorTicker.append(Ticker)
        continue   
        


pd.Series(ErrorTicker).to_excel("./Tiingo_Stock_10min/" + "ErrorTicker_10min.xlsx", index=False)

pd.Series(iTicker).to_excel("./Tiingo_Stock_10min/" + "iTicker_10min.xlsx", index=False)

# if __name__=='__main__':
#     main()     
    

# aa = pd.Series(ErrorTicker)
# aa.to_excel('ErrorTicker.xlsx')


