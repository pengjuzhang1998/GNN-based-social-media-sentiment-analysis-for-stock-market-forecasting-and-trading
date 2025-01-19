import pickle
from textblob import TextBlob
import numpy as np
from langdetect import detect

def find_language(text):
    try:
        language = detect(text)
    except:
        # langdetect cannot detect the language, use a fallback language
        language = 'unknown'
    return language


def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return polarity
""" core_author_dict = []
with open('2017-01-01T09_00_00ZXOM_dict.pkl', 'rb') as f:
    data = pickle.load(f) """

def sentiment_rate(data , core_author_dict = {}):
    data = data['data']
    sentiment_rate = []
    for twi in data:
        polarity = get_sentiment(twi['text'])
        if twi['author_id'] in core_author_dict:
            polarity = (core_author_dict[twi['author_id']] + 1) *  polarity
        if find_language(twi['text']) == 'en':
            sentiment_rate.append(polarity)

    sentiment_rate_mean = sum(sentiment_rate) / len(sentiment_rate)
    return sentiment_rate_mean

#print(sentiment_rate(data))