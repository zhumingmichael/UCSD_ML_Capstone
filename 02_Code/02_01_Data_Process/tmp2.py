%reset -f
from pandas_datareader import data
import pandas as pd

symbols = ['AAPL', 'GOOG']
stock_data = {symbol: data.get_data_yahoo(symbol) for symbol in symbols}

meltAll = []

for iTicker in range(len(symbols)):
    # iTicker= 0
    Ticker = symbols[iTicker]
    
    df = stock_data[Ticker]
    df = df.tail(1000).copy()
    df['time'] = df.index
    df['symbol'] = Ticker
    
    melted = df.melt(id_vars=['symbol','time'])
    # melted.rename(columns={'variable':'kind', 'value':'price'}, inplace=True)
    # melted['time'] = list(df.index) * len(df.columns)
    # melted['symbol'] = 'AAPL'
    
    if(len(meltAll)<=0):
        meltAll = melted
    else:
        meltAll = pd.concat([meltAll, melted], axis = 0)




from tsfresh.utilities.dataframe_functions import roll_time_series
use_data_of = 100

rolled = roll_time_series(meltAll.copy(), 
                           column_id = 'symbol', 
                          column_sort = 'time',
                           # column_kind='variable', 
                          # rolling_direction=1,
                          max_timeshift=use_data_of-1,
                          min_timeshift=use_data_of-1)


# .reset_index(drop=True)
set(melted.symbol)
set(meltAll.symbol)
set(rolled.symbol)

# rolled2 = rolled.copy()

# rolled = rolled.groupby('id').filter(lambda x: len(set(x['time']))==use_data_of).reset_index(drop=True)

del df, meltAll, melted


from tsfresh import extract_features
from tsfresh.feature_extraction.settings import EfficientFCParameters

X = extract_features(rolled, column_id = 'id', column_sort='time',
                     column_kind='variable', column_value='value',
                     default_fc_parameters = EfficientFCParameters(), n_jobs=4)

XX = X.copy()
X1 = XX.iloc[1,:]
XX.dropna()
dd = X.apply(lambda x: sum(x.isna()))

X.columns
XX = X.T.iloc[:,0].copy()
