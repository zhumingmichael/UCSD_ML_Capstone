%reset -f

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute

try:
    import pandas_datareader.data as web
except ImportError:
    print("You need to install the pandas_datareader. Run pip install pandas_datareader.")

from sklearn.linear_model import LinearRegression

## Reading the data

# We download the data from "stooq" and only store the High value.
# Please note: this notebook is for showcasing `tsfresh`s feature extraction - not to predict stock market prices :-)

df = web.DataReader("AAPL", 'stooq')["High"]
df.head()

plt.figure(figsize=(15, 6))
df.plot(ax=plt.gca())
plt.show()

# We want to make the time dependency a bit clearer and add an identifier to each of the stock values (in this notebook we only have Google though).

df_melted = pd.DataFrame({"high": df.copy()})
df_melted["date"] = df_melted.index
df_melted["Symbols"] = "AAPL"

df_melted.head()

## Create training data sample

# Forecasting typically involves the following steps:
# * take all data up to today
# * do feature extraction (e.g. by running `extract_features`)
# * run a prediction model (e.g. a regressor, see below)
# * use the result as the forecast for tomorrow

# In training however, we need multiple examples to train.
# If we would only use the time series until today (and wait for the value of tomorrow to have a target), we would only have a single training example.
# Therefore we use a trick: we replay the history.

# Imagine you have a cut-out window sliding over your data.
# At each time step $t$, you treat the data as it would be today. 
# You extract the features with everything you know until today (which is all data until and including $t$).
# The target for the features until time $t$ is the time value of time $t + 1$ (which you already know, because everything has already happened).

# The process of window-sliding is implemented in the function `roll_time_series`.
# Our window size will be 20 (we look at max 20 days in the past) and we disregard all windows which are shorter than 5 days.

df_rolled = roll_time_series(df_melted, column_id="Symbols", column_sort="date",
                             max_timeshift=20, min_timeshift=5)

df_rolled.head()

# The resulting dataframe now consists of these "windows" stamped out of the original dataframe.
# For example all data with the `id = (AAPL, 2020-07-14 00:00:00)` comes from the original data of stock `AAPL` including the last 20 days until `2020-07-14`:

df_rolled[df_rolled["id"] == ("AAPL", pd.to_datetime("2020-07-14"))]

df_melted[(df_melted["date"] <= pd.to_datetime("2020-07-14")) & 
          (df_melted["date"] >= pd.to_datetime("2020-06-15")) & 
          (df_melted["Symbols"] == "AAPL")]

# If you now group by the new `id` column, each of the groups will be a certain stock symbol until and including the data until a certain day (and including the last 20 days in the past).

# Whereas we started with 1259 data samples:

len(df_melted)

# we now have 1254 unique windows (identified by stock symbol and ending date):

df_rolled["id"].nunique()

# We "lost" 5 windows, as we required to have a minimum history of more than 5 days.

df_rolled.groupby("id").size().agg([np.min, np.max])

# The process is also shown in this image (please note that the window size is smaller for better visibility):

# <img src="./stocks.png"/>

## Extract Features

# The rolled (windowed) data sample is now in the correct format to use it for `tsfresh`s feature extraction.
# As normal, features will be extracted using all data for a given `id`, which is in our case all data of a given window and a given id (one colored box in the graph above).

# If the feature extraction returns a row with the index `(AAPL, 2020-07-14 00:00:00)`, you know it has been calculated using the `AAPL` data up and including `2020-07-14` (and 20 days of history).

X = extract_features(df_rolled.drop("Symbols", axis=1), 
                     column_id="id", column_sort="date", column_value="high", 
                     impute_function=impute, show_warnings=False)

X.head()
# del df, df_melted, df_rolled
# We make the data a bit easier to work with by removing the tuple-index

X = X.set_index(X.index.map(lambda x: x[1]), drop=True)
X.index.name = "last_date"
X.head()

# Our `(AAPL, 2020-07-14 00:00:00)` is also in the data again:

X.loc['2020-07-14']


## Prediction
y = df_melted.set_index("date").sort_index().high.shift(-1)

y["2020-07-13"], df["2020-07-14"].iloc[0]



y = y[y.index.isin(X.index)]
X = X[X.index.isin(y.index)]

X[:"2018"]


X_train = X[:"2018"]
X_test = X["2019":]

y_train = y[:"2018"]
y_test = y["2019":]


X_train_selected = select_features(X_train, y_train)

ada = LinearRegression()

ada.fit(X_train_selected, y_train)

# Now lets check how good our prediction is:

X_test_selected = X_test[X_train_selected.columns]

y_pred = pd.Series(ada.predict(X_test_selected), index=X_test_selected.index)

# The prediction is for the next day, so for drawing we need to shift 1 step back:

plt.figure(figsize=(15, 6))

y.plot(ax=plt.gca())
y_pred.plot(ax=plt.gca(), legend=None, marker=".")













