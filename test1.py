import pandas as pd
import numpy as np
from fbprophet import Prophet
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
import datetime

start = datetime.datetime(2021, 1, 1)
end = datetime.datetime(2021, 10, 18)

data = web.DataReader("F", 'yahoo', start, end)

data.plot()


