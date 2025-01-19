import pickle
from relation import get_relation
from core_account import core_account
import time
from data_preprocessing import traverse_files
import torch


bearer_token = 'AAAAAAAAAAAAAAAAAAAAAGikQgEAAAAANJ0LNIV3UMxtrftSQf5Zw1FsOws%3Df5c95trByZYotJdN6R0ud3VkzEdzhoaxc4Ekb5aXVvOeBBoxSk'


def get_coomat(user_dict):  
    account_list=[]      
    for user in user_dict:
        account_list.append(user)
    return account_list

        
def set_coomat(user_dict):   
    account_list = get_coomat(user_dict)    
    coomat = []
    n=0
    acc_sym = {} ##The symple of the account in coomat
    for account in account_list:
        acc_sym[account] = n
        n = n+1
    for i in range(len(account_list)):
        for j in range(len(account_list)):
            coomat.append([[i, j] , 0])

    for user in user_dict:
        data = user_dict[user]
        s1 = acc_sym[user]
        for account_to, connect in data.inside_connection.items():
            c=0
            if account_to in acc_sym:
                s2 = acc_sym[account_to]
            else:
                continue
            if connect == [1,0]:
                c=1
            elif connect ==[0,1]:
                c=1
            elif connect == [1,1]:
                c=2
            if coomat[s1 * len(account_list) + s2][0] == [s1,s2]:
                coomat[s1 * len(account_list) + s2][1] = c
                        #maybe change into coomat[s1 * len(account_list) + s2][1] += c later as well as change the type of c
    row_indices = torch.tensor([entry[0][0] for entry in coomat])
    col_indices = torch.tensor([entry[0][1] for entry in coomat])
    values = torch.tensor([entry[1] for entry in coomat])
    coo_matrix = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]), values)

    return coo_matrix, acc_sym
                    

class Tweets:
    def __init__(self, id, text, author):
        self.id = id
        self.text = text
        self.author = author
        self.relation_matrix = get_relation(self.id, bearer_token)
    def likes(self):
        return self.relation_matrix['likes']
    def retweets(self):
        return self.relation_matrix['retweets']
class User:
    def __init__(self, id):
        self.id = id
        self.tweets = []
        self.inside_connection = {}
        self.outside_connection = [0,0]

#dir_path = './'
#end_time = '2022-12-31'
def set_graph(dir_path, end_time, stock_name, time_period):
    data = traverse_files(dir_path = dir_path, end_time = end_time, time_period = time_period , stock_name = stock_name)
    users_list=[]
    tweets_list = []
    core_account_list = core_account(data, matrix_limit=50)
    mainly_tweets_limit = 10 
    core_user_dict={}
    for users, user_data in data.items():
        if users in core_account_list:
            account = User(users)
            for twis in user_data['twis']:
                #print(twis)
                if twis['public_metrics']['like_count']>=mainly_tweets_limit and twis['public_metrics']['retweet_count']>=mainly_tweets_limit:
                    test_data = Tweets(twis['id'], twis['text'], twis['author_id'])
                    if test_data.relation_matrix == 'Error':
                        time.sleep(900)
                        test_data = Tweets(twis['id'], twis['text'], twis['author_id'])

                    account.tweets.append(test_data)
        
            core_user_dict[users]=account
    core_user = {}
    for users, user_data in core_user_dict.items():
        twis = user_data.tweets
        if len(twis) == 0:
            continue
        in_connect = {}
        #print(twis)
        for twi in twis:
            likes= twi.relation_matrix['likes']
            retweets= twi.retweets()
            for likes in twi.likes():
                if likes['id'] in core_account_list:
                    if likes['id'] not in in_connect:
                        in_connect[likes['id']] = [0,0]
                    in_connect[likes['id']][0] = in_connect[likes['id']][0] + 1
                else:
                    user_data.outside_connection[0] = user_data.outside_connection[0] + 1
            for retweets in twi.retweets():
                if retweets['id'] in core_account_list:
                    if retweets['id'] not in in_connect:
                        in_connect[retweets['id']] = [0,0]
                    in_connect[retweets['id']][1] = in_connect[retweets['id']][1] + 1
                else:
                    user_data.outside_connection[1] = user_data.outside_connection[1] + 1
        user_data.inside_connection = in_connect
        core_user[users] = user_data

    rel_matrix, node_dict = set_coomat(core_user)
    return rel_matrix, node_dict
