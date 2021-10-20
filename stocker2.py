# basic
import numpy as np
import pandas as pd

# get data
import pandas_datareader as pdr

# visual
import matplotlib.pyplot as plt
# matplotlib inline

#time
import datetime as datetime

#Prophet
from fbprophet import Prophet


from sklearn import metrics
start = datetime.datetime(2015,1,5)
df_2492 = pdr.DataReader('2492.TW', 'yahoo', start=start)
plt.style.use('ggplot')
df_2492['Adj Close'].plot(figsize=(12, 8))

new_df_df_2492['y'] = np.log(new_df_df_2492['y'])
# 定義模型
model = Prophet()

# 訓練模型
model.fit(new_df_df_2492)

# 建構預測集
future = model.make_future_dataframe(periods=365) #forecasting for 1 year from now.

# 進行預測
forecast = model.predict(future)

figure=model.plot(forecast)

df_2492_close = pd.DataFrame(df_2492['Adj Close'])
two_years = forecast.set_index('ds').join(df_2492_close)
two_years = two_years[['Adj Close', 'yhat', 'yhat_upper', 'yhat_lower' ]].dropna().tail(800)
two_years['yhat']=np.exp(two_years.yhat)
two_years['yhat_upper']=np.exp(two_years.yhat_upper)
two_years['yhat_lower']=np.exp(two_years.yhat_lower)
two_years[['Adj Close', 'yhat']].plot(figsize=(8, 6));

two_years_AE = (two_years.yhat - two_years['Adj Close'])
two_years_AE.describe()

print ("MSE:",metrics.mean_squared_error(two_years.yhat, two_years['Adj Close']))