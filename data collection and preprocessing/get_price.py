import yfinance as yf
import pandas as pd
import numpy as np


stock_symbol = 'BTC-USD' 
start_date = '2016-01-01' 
end_date = '2022-12-31'


stock_data = yf.download(stock_symbol, start=start_date, end=end_date)


N = 30
stock_data['Daily_Returns'] = stock_data['Close'].pct_change()
stock_data['Volitality'] = stock_data['Daily_Returns'].rolling(N).std()
stock_data['Volitality'].iloc[:N-1] = None
stock_data.to_csv(stock_symbol+'_data.csv')

#XOM: VUSA.L, BLK, STT
#AAPL: VUSA.L, BLK, BRK-A