import os
import glob
import pickle
from text_analyse import sentiment_rate
import pandas as pd
from account_rate import sort_influence_rate
from datetime import datetime, timedelta

stock_list = ['JPM', 'XOM', 'JNJ']
for stock_name in stock_list:
    dir_path = './data/'+ stock_name +'/'
    account_influence_rate = sort_influence_rate(influence_rate = 'influence_rate_'+stock_name+'.pkl', node= 'node_'+stock_name+'.pkl')
    end_time = '2022-12-31'
    time_period = 365


    end_date = datetime.strptime(end_time, "%Y-%m-%d")
    date_list = [end_date - timedelta(days=i) for i in range(time_period)]
    date_list = [date.strftime("%Y-%m-%d") for date in date_list]


    data_frame = pd.DataFrame(columns=['label', 'value'])

    for date in date_list:
        sub_dir_path = os.path.join(dir_path, date)
        daily_sentiment = []
        for root, dirs, files in os.walk(sub_dir_path):
            for file_name in files:
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        sentiment_day = sentiment_rate(data, account_influence_rate)
                        daily_sentiment.append(sentiment_day)
        daily_sentiment_mean = sum(daily_sentiment)/len(daily_sentiment)
        data_frame = data_frame.append({'label': sub_dir_path, 'value': daily_sentiment_mean}, ignore_index=True)
    """ for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            pkl_files = glob.glob(os.path.join(dir_path, '*.pkl'))
            daily_sentiment = []
            for pkl_file in pkl_files:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f) 
                    sentiment_day = sentiment_rate(data, account_influence_rate)
                    daily_sentiment.append(sentiment_day)
            daily_sentiment_mean = sum(daily_sentiment)/len(daily_sentiment)
            data_frame = data_frame.append({'label': dir_path[8:], 'value': daily_sentiment_mean}, ignore_index=True)
    """
    data_frame.to_csv('data_withrate'+stock_name+'.csv', index=False)