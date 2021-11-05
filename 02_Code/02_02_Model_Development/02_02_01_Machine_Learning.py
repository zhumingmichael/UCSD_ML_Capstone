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

# Get the file names of the data
filesnames = listdir(datapath)
# Extract the tickers from the file names
AvaTickers = [x.split("_daily.xlsx")[0] for x in filesnames]
AvaTickers = np.sort(AvaTickers)

### Data transformation #33
Ticker = 'LSCC'
Sdata = pd.read_csv(datapath + r"\\" + Ticker + "_daily.csv") 
Sdata.columns
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


# Sdata = Sdata.tail(-1)

## Interpolate NAs ##
Sdata = Sdata.interpolate(method='linear', limit_direction='forward', axis=0)
# Sdata = Sdata.dropna()
Sdata['Ticker'] = Ticker

### tsfresh generation ###
Sdata.columns


df = Sdata[['Ticker', 'date', 'volume_log', 'adjClose_log','HL_Log_Spread', 'OS_Log_Spread',\
            'Ret_D','Vol_D', 'Ret_W', 'Ret_M', 'Vol_W', 'Vol_M']].copy()
# df = Sdata[['Ticker', 'date', 'Ret_D']].copy()

df['date'] = [pd.to_datetime(x) for x in df['date']]
# df = df.tail(100 + 21*3)

df.dropna(inplace=True)
del Sdata

from tsfresh.utilities.dataframe_functions import roll_time_series
use_data_of = 21*3
melted = df.melt(id_vars=['Ticker','date'])
melted.columns
rolled = roll_time_series(melted.copy(), 
                          column_id = 'Ticker', 
                          column_sort = 'date',
                          column_kind='variable', 
                          rolling_direction=1,
                          max_timeshift=use_data_of-1,
                          min_timeshift=use_data_of-1).reset_index(drop=True)

rolled.columns
set(rolled.Ticker)
set(rolled.date)
set(rolled.variable)
set(rolled.value)
set(rolled.id)

rolled[rolled["id"] == ("LSCC", pd.to_datetime("2020-07-14"))]
rolled["id"].nunique()
rolled.groupby("id").size().agg([np.min, np.max])
del AvaTickers, melted


from tsfresh import extract_features
from tsfresh.feature_extraction.settings import EfficientFCParameters

X = extract_features(rolled, column_id = 'id', column_sort='date',
                      column_kind='variable', column_value='value',
                      default_fc_parameters = EfficientFCParameters(), n_jobs=4)



AllNa_Cols = X.apply(lambda x: sum(x.isna()) >= X.shape[0]*0.01, axis=0)
X = X.loc[:,~AllNa_Cols.values].copy()
X = X.dropna()

# X = pd.read_pickle(savepath + r"\\" + Ticker + r"_tsfresh.zip")


X.to_pickle(savepath + r"\\" + Ticker + r"_tsfresh.zip")
X = pd.read_pickle(savepath +r"\Ret_D_tsfresh.zip")

##### Model #####
## Prediction
X = X.set_index(X.index.map(lambda x: x[1]), drop=True)

y_Ret_D = df.set_index("date").sort_index().Ret_D.shift(-1).copy()
y_Vol_D = df.set_index("date").sort_index().Vol_D.shift(-1).copy()
y_Ret_W = df.set_index("date").sort_index().Ret_W.shift(-5).copy()
y_Vol_W = df.set_index("date").sort_index().Vol_W.shift(-5).copy()
y_Ret_M = df.set_index("date").sort_index().Ret_M.shift(-21).copy()
y_Vol_M = df.set_index("date").sort_index().Vol_M.shift(-21).copy()


y = df.set_index("date").sort_index().UCTT_Ret_D.shift(-1).copy()

y = y[y.index.isin(X.index)]
X = X[X.index.isin(y.index)]

X_train = X[:"2019"]
X_test = X["2020":]

y_train = y[:"2019"]
y_test = y["2020":]


# X_train_selected = select_features(X_train, y_train)
X_train_selected = X_train
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ada = RandomForestRegressor(n_estimators = 100, max_depth = 7, n_jobs=4, )

ada.fit(X_train_selected, y_train)

# Now lets check how good our prediction is:

X_test_selected = X_test[X_train_selected.columns]

y_pred = pd.Series(ada.predict(X_test_selected), index=X_test_selected.index)

# The prediction is for the next day, so for drawing we need to shift 1 step back:
y_true = y.shift(1)
plt.figure(figsize=(15, 6))

y_true.plot(ax=plt.gca())
y_pred.plot(ax=plt.gca(), legend=None, marker=".")

y_true = y_true[y_true.index.isin(y_pred.index)]

from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred,  squared=False)

Ret_true_cum = np.exp(y_true.cumsum())
Ret_pred_cum = np.exp(y_pred.cumsum())

plt.figure(figsize=(15, 6))

y_true.plot(ax=plt.gca())
y_pred.plot(ax=plt.gca(), legend=None, marker=".")


plt.figure(figsize=(15, 6))

Ret_true_cum.plot(ax=plt.gca())
Ret_pred_cum.plot(ax=plt.gca(), legend=None, marker=".")

#### Weekly ####
## Return ##
y = y_Ret_W

y = y[y.index.isin(X.index)]
X = X[X.index.isin(y.index)]

X_train = X[:"2019"]
X_test = X["2020":]

y_train = y[:"2019"]
y_test = y["2020":]


# X_train_selected = select_features(X_train, y_train)
X_train_selected = X_train
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ada = RandomForestRegressor(n_estimators = 100, max_depth = 7, n_jobs=4, )

ada.fit(X_train_selected, y_train)

# Now lets check how good our prediction is:

X_test_selected = X_test[X_train_selected.columns]

y_pred = pd.Series(ada.predict(X_test_selected), index=X_test_selected.index)

# The prediction is for the next day, so for drawing we need to shift 1 step back:
y_true = y.shift(5)
plt.figure(figsize=(15, 6))

y_true.plot(ax=plt.gca())
y_pred.plot(ax=plt.gca(), legend=None, marker=".")

y_true = y_true[y_true.index.isin(y_pred.index)]

from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred,  squared=False)

Ret_true_cum = np.exp(y_true.cumsum())
Ret_pred_cum = np.exp(y_pred.cumsum())

plt.figure(figsize=(15, 6))

y_true.plot(ax=plt.gca())
y_pred.plot(ax=plt.gca(), legend=None, marker=".")


## Volatility ##
y = y_Vol_W

y = y[y.index.isin(X.index)]
X = X[X.index.isin(y.index)]

X_train = X[:"2019"]
X_test = X["2020":]

y_train = y[:"2019"]
y_test = y["2020":]


# X_train_selected = select_features(X_train, y_train)
X_train_selected = X_train
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ada = RandomForestRegressor(n_estimators = 100, max_depth = 7, n_jobs=4, )

ada.fit(X_train_selected, y_train)

# Now lets check how good our prediction is:

X_test_selected = X_test[X_train_selected.columns]

y_pred = pd.Series(ada.predict(X_test_selected), index=X_test_selected.index)

# The prediction is for the next day, so for drawing we need to shift 1 step back:
y_true = y.shift(5)
plt.figure(figsize=(15, 6))

y_true.plot(ax=plt.gca())
y_pred.plot(ax=plt.gca(), legend=None, marker=".")

y_true = y_true[y_true.index.isin(y_pred.index)]

from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred,  squared=False)

Ret_true_cum = np.exp(y_true.cumsum())
Ret_pred_cum = np.exp(y_pred.cumsum())

plt.figure(figsize=(15, 6))

y_true.plot(ax=plt.gca())
y_pred.plot(ax=plt.gca(), legend=None, marker=".")













