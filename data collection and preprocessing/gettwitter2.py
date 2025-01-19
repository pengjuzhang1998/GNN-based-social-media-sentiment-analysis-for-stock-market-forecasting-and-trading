import requests
import os
import json
import pandas as pd
import datetime
import time
import pickle
from creat_timelist import creat_timelist


bearer_token = 'AAAAAAAAAAAAAAAAAAAAAGikQgEAAAAANJ0LNIV3UMxtrftSQf5Zw1FsOws%3Df5c95trByZYotJdN6R0ud3VkzEdzhoaxc4Ekb5aXVvOeBBoxSk'

search_url = "https://api.twitter.com/2/tweets/search/all?tweet.fields=public_metrics&expansions=author_id&user.fields=created_at"

stock_list=['AAPL', 'XOM', 'JNJ', 'JPM']
stock_dict={'AAPL':'Apple Inc.', 'XOM':'Exxon Mobil Corporation', 'JNJ':'Johnson & Johnson', 'JPM':'JPMorgan Chase & Co.'}
#AAPL	Apple Inc.
#XOM	Exxon Mobil Corporation
#JNJ	Johnson & Johnson
#JPM	JPMorgan Chase & Co.
def create_headers(bearer_token):
    headers = {
        "Authorization": "Bearer {}".format(bearer_token),
        "User-Agent": "v2SpacesSearchPython"
    }
    return headers


def connect_to_endpoint(url, headers, params):
    response = requests.request("GET", search_url, headers=headers, params= params)
    #print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    if response.status_code == 200:    
        return response.json()
    else:
        return response.status_code


def main():
    #creat timelist
    start_date = datetime.datetime(2021, 9, 28, 15, 0, 0)
    end_date = datetime.datetime(2023, 1, 1, 0, 0, 0)
    time_list=creat_timelist(start_date, end_date, 3)
    start_time= '2021-09-28T12:00:00Z'
    for the_time in time_list:
        end_time = the_time
        for stock in stock_list:
            data = search_tweets(stock,start_time, end_time)
            #print(type(data))
            with open(start_time+stock+"_dict.pkl", "wb") as f:
                pickle.dump(data, f)
                print(the_time)
            time.sleep(5.1)
        start_time = the_time
    

   

def search_tweets(search_term, start_time, end_time):
    headers = create_headers(bearer_token)
    query_params = {'max_results': 500,'start_time': start_time, 'end_time': end_time, 'query': search_term}
    json_response = connect_to_endpoint(search_url, headers, query_params)
    time.sleep(5.1)
    query_params = {'max_results': 500,'start_time': start_time, 'end_time': end_time, 'query': stock_dict[search_term]}
    json_response_full = connect_to_endpoint(search_url, headers, query_params)
    json_response = json_response+json_response_full
    if type(json_response) == dict:
        return json_response
    else:
        print(json_response)
        return 'NaN'
    

if __name__ == "__main__":
    main()