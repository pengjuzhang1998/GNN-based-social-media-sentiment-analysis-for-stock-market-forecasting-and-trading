import requests
import json
import pickle
import os
import time

# Bearer Token
#bearer_token = 'AAAAAAAAAAAAAAAAAAAAAGikQgEAAAAANJ0LNIV3UMxtrftSQf5Zw1FsOws%3Df5c95trByZYotJdN6R0ud3VkzEdzhoaxc4Ekb5aXVvOeBBoxSk'

#tweet_id = "815388372919140353"

def get_relation(tweet_id, bearer_token):
    filename = "twi_data"+tweet_id+".pkl"
    current_direc = os.getcwd()
    dir_path = os.path.join(current_direc, 'twi_')
    file_path = os.path.join(dir_path, filename)

    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Set endpoint URLs
        twi_relationship={'retweets':[], 'likes':[]}
        retweets_url = f"https://api.twitter.com/2/tweets/{tweet_id}/retweeted_by"
        likes_url = f"https://api.twitter.com/2/tweets/{tweet_id}/liking_users"

        
        # Set headers
        headers = {"Authorization": f"Bearer {bearer_token}"}

        # Send GET requests
        retweets_response = requests.get(retweets_url, headers=headers)
        likes_response = requests.get(likes_url, headers=headers)
        
        # Print the usernames of retweeters
        if retweets_response.status_code == 200:
            retweets = json.loads(retweets_response.text)
            #print(retweets)
            if "data" in retweets:
                twi_relationship['retweets'] = retweets['data']

        else:
            print(f"Error: {retweets_response.status_code} - {retweets_response.reason}")
            return 'Error'

        # Print the usernames of likers
        if likes_response.status_code == 200:
            likes = json.loads(likes_response.text)
            if "data" in likes:
                twi_relationship['likes'] = likes['data']
        else:
            print(f"Error: {likes_response.status_code} - {likes_response.reason}")
            return 'Error'


        with open("twi_data"+tweet_id+".pkl", "wb") as f:
            pickle.dump(twi_relationship, f)
            print('save')
        time.sleep(5.1)
        return twi_relationship

#print(get_relation(tweet_id, bearer_token))