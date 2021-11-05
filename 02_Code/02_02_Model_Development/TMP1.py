
y_ret = df.set_index("date").sort_index().UCTT_Ret_D.shift(-1).copy()
y = y_ret.copy()
y[y_ret-y_ret.mean() >= 2*y_ret.std()] = 'E'
y[(y_ret-y_ret.mean() >= 1*y_ret.std()) & (y_ret-y_ret.mean() <= 2*y_ret.std())] = 'D'
y[(y_ret-y_ret.mean() >= -1*y_ret.std()) & (y_ret-y_ret.mean() <= 1*y_ret.std())] = 'C'
y[(y_ret-y_ret.mean() <= -1*y_ret.std()) & (y_ret-y_ret.mean() >= -2*y_ret.std())] = 'B'
y[y_ret-y_ret.mean() <= -2*y_ret.std()] = 'A'


y = y[y.index.isin(X.index)]
X = X[X.index.isin(y.index)]

X_train = X[:"2019"]
X_test = X["2020":]

y_train = y[:"2019"]
y_test = y["2020":]


# X_train_selected = select_features(X_train, y_train)
X_train_selected = X_train
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

ada = RandomForestRegressor(n_estimators = 100, max_depth = 7, n_jobs=4 )

ada.fit(X_train_selected, y_train)


clf = RandomForestClassifier(n_estimators = 100, max_depth = 6, n_jobs=4, criterion='Entropy')
clf.fit(X_train_selected, y_trian)

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
