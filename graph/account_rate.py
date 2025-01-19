import pickle
def sort_influence_rate(influence_rate = 'influence_rate.pkl', node = 'node.pkl'):
    with open(influence_rate, 'rb') as f:
        rate = pickle.load(f)

    with open(node, 'rb') as f:
        data = pickle.load(f)

    core_account_dict = {}
    for account, label in data.items():
        if label < len(rate):
            core_account_dict[account] = rate[label, label]

    return core_account_dict