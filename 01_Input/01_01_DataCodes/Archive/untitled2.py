Ticker = '000012'

reqdata = requests.get("https://api.tiingo.com/tiingo/daily/"+Ticker+"/"+\
                "prices?startDate=2020-01-01&"+\
                # "resampleFreq=1day&"+\
                "columns=open,high,low,close,volume,adjOpen,adjHigh,adjLow,adjClose,adjVolume,divCash,splitFactor&"+\
                "token="+Token, headers=headers)
    
    
NewData = pd.DataFrame(reqdata.json())
