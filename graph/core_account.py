import pickle


def count_public(twi):
    pub_m= twi['public_metrics']
    num = pub_m['retweet_count'] + pub_m['like_count']
    return num

def core_account(data, matrix_limit=50):
    core_account_list = []
    for users, user_data in data.items():
        #print(users)
        user_num=0
        for twis in user_data['twis']:
            user_num = user_num + count_public(twis)
        if user_num >matrix_limit:
            core_account_list.append(users)
    return core_account_list