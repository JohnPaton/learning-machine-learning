import pandas as pd
import numpy as np
import quandl, datetime, pickle, time
import os
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import regression_alg as ra

style.use('ggplot')

def get_data():
    # pickle today's dataset so we don't use all the free gets
    d_pickle = 'WIKI_GOOGL'+str(time.strftime("%d%m%Y"))+'.pickle'
    if os.path.isfile(d_pickle): 
        df = pd.read_pickle(d_pickle)
    else:
        df = quandl.get('WIKI/GOOGL')
        df.to_pickle(d_pickle)

    df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0

    df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']] # Just keep useful features
    return df

df = get_data()

forecast_col = 'Adj. Close'
df = df.fillna(-99999)

# predict 30 days into the future
forecast_out = 30 #math.ceil(0.01*len(df))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
# X = preprocessing.scale(X)
X_lately = X[-forecast_out:] # most recent features have no label
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

# bootstrap data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

# get sklearn model
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train,y_train)
sk_accuracy = clf.score(X_test,y_test)
sk_forecast_set = clf.predict(X_lately)

# get our own model
beta = ra.fit_multi(X_train, y_train)
alg_forecast_set = ra.predict_multi(X_lately,beta)
alg_accuracy = ra.r2(y_test, ra.predict_multi(X_test,beta))

# Compare accuracy (should be extremely similar)
print("sk-learn:",sk_accuracy,"\tOwn alg:",alg_accuracy)


df['SK_Forecast'] = np.nan
df['ALG_Forecast'] = np.nan

# hack date to insert predictions
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in range(len(sk_forecast_set)):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-2)]\
                        + [sk_forecast_set[i], alg_forecast_set[i]]


# plot
df['Adj. Close'].plot()
df['SK_Forecast'].plot()
df['ALG_Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





#print(accuracy)






