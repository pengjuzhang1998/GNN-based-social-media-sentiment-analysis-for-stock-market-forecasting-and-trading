import pandas as pd

stocks = ['MSFT', 'GOOGL', 'AMZN', 'ABT', 'MRK', 'AMGN', 'BAC', 'PNC', 'C', 'CVX', 'COP', 'HAL']

for name in stocks:
    data = pd.read_csv(name+'_data.csv', index_col='Date', parse_dates=True)

    date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    data_reindexed = data.reindex(date_range)


    data_filled = data_reindexed.fillna(method='ffill')

    data_filled.to_csv(name+'_data_fill.csv')
