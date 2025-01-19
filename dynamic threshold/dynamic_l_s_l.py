import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import levy
import warnings
import math
import random
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
class Funding:
    def __init__(self, H, price):
        self.H = H
        self.price = price

    def sell(self, t_cost):
        return self.price * (1-t_cost)
    
    def hold(self, daily_return):
        return self.price * (1+daily_return)
    
    def buy(self, t_cost, daily_return):
        return self.price * (1-t_cost) * (1+daily_return)

def sample_normal_below_n(n, std=None):
    if n==0:
        return 0
    if std is None:
        std = n / 6
    
    for i in range(1000):
        sample = np.random.normal(loc=0, scale=std)
        if 0 <= sample <= n:
            return n-int(sample)

def sample_levy_below_n(n, loc=0):
    if n == 0:
        return 0
    for i in range(1000):
        sample = levy.rvs(loc = loc, scale = n/3 )
        if 0 <= sample <= n:
            return n-int(sample)
def trading(p_dailyreturn, a_dailyreturn, funding, T_posi, T_nega, t_cost, acc_matrix = np.zeros((2,3)), t_nums = 0):
    if funding.H == True:
        if p_dailyreturn > T_nega:
            funding.price = funding.hold(a_dailyreturn)
            if p_dailyreturn > T_posi:
                if a_dailyreturn > 0:
                    acc_matrix[0,0] = acc_matrix[0,0] + 1
                elif a_dailyreturn < 0:
                    acc_matrix[1,0] = acc_matrix[1,0] + 1
            else:
                if a_dailyreturn > 0:
                    acc_matrix[0,1] = acc_matrix[0,1] + 1
                elif a_dailyreturn < 0:
                    acc_matrix[1,1] = acc_matrix[1,1] + 1 
        else:
            funding.price = funding.sell(t_cost)
            t_nums = t_nums + 1
            if a_dailyreturn > 0:
                acc_matrix[0,2] = acc_matrix[0,2] + 1
            elif a_dailyreturn < 0:
                acc_matrix[1,2] = acc_matrix[1,2] + 1
            funding.H = False
    else:
        if p_dailyreturn > T_posi:
            funding.price = funding.buy(t_cost, a_dailyreturn)
            t_nums = t_nums + 1
            funding.H = True
            if a_dailyreturn > 0:
                acc_matrix[0,0] = acc_matrix[0,0] + 1
            elif a_dailyreturn < 0:
                acc_matrix[1,0] = acc_matrix[1,0] + 1
        else:
            funding.price = funding.price
            if p_dailyreturn > T_nega:
                if a_dailyreturn > 0:
                    acc_matrix[0,1] = acc_matrix[0,1] + 1
                elif a_dailyreturn < 0:
                    acc_matrix[1,1] = acc_matrix[1,1] + 1 
            else:
                if a_dailyreturn > 0:
                    acc_matrix[0,2] = acc_matrix[0,2] + 1
                elif a_dailyreturn < 0:
                    acc_matrix[1,2] = acc_matrix[1,2] + 1
    return acc_matrix, t_nums

def t_data(window_size, small_window, i, size):
    x=[]
    y=[]
    for j in range(window_size-small_window-1):  
        x.append(data[i+j:i+j+small_window, :])  # use all 9 columns as input
        y.append(data[i+j+small_window, :size]) 
    x = np.array(x)
    y = np.array(y)
                #  Transform to tensors
    x = torch.tensor(x, dtype=torch.float32)  # Shape: batch_size, sequence_length, num_features
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: batch_size, 1
    return x,y
stocks_name = [ 'XOM']
sentiment_type = ['nosentiment', 'noinfluence', 'withinfluence']
related_type = ['related','norelated', 'norelated_3']
random_method_list = ['random', 'normal','levy']
window = [5, 15, 30]
t_cost = 0.002
base_t = 0.01
small_window = 5
hidden_dim = 64
num_epochs = 70
large_window = 15
class LSTMModel(nn.Module):
                    
    def __init__(self, ninput, hidden_dim, noutput, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(ninput, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, noutput)
                    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

for random_method in random_method_list:

    for stock_name in stocks_name:
        for window_size in window:
            for relate in related_type:
                for senti in sentiment_type:

                    if relate == 'related':
                        file_path = 'LSTM/LSTM_predict_' + stock_name + '_' + senti + str(window_size) + '.csv'
                    else:
                        file_path = 'LSTM/LSTM_predict_' + stock_name + '_' + senti + '_' + relate + str(window_size) + '.csv'
                    csv_name= './result_gpu/sharpe/'+random_method+'/'+file_path
                    if os.path.exists(csv_name):
                        continue
                    df = pd.read_csv(file_path)

                    p_d = df['Predicted_Daily_Returns'].tail(2000).to_numpy()
                    a_d = df['Actual_Daily_Returns'].tail(2000).to_numpy()
                    file_path_all = './' + stock_name + '_data_fill.csv'
                    df_full = pd.read_csv(file_path_all)
                    vol = df_full['Volatility'].tail(2000).to_numpy()
                    
                    senti_influence = df_full['Sentiment with influence'].tail(2000).to_numpy()
                    
                    volume = df_full['Volume'].tail(2000).to_numpy()

                    best_threshold = np.zeros(2000)
                    for i in range(len(p_d)):
                        best_threshold[i] = abs(a_d[i]/vol[i]) * (1-np.mean(df['Actual_Daily_Returns'][-2000-30+i:-2000+i]))
                    sharpe_r = p_d/vol
                    
                    data = np.array([sharpe_r, best_threshold, vol, senti_influence, volume]).T
               
                    
                    model = LSTMModel(ninput=small_window, hidden_dim=hidden_dim, noutput=small_window)
                    model = model.to(device)
                    scaler = MinMaxScaler()
                    data = scaler.fit_transform(data)
                  
                    
                    x_old=[]
                    predicted_data = np.zeros((len(data),5))
                    actual_data = np.zeros((len(data),5))
                    for i in range(len(data) - large_window - 1):
                        for n in range(int(len(x_old) /200)):
                            if x_old != []:
                                if random_method == 'normal':
                                    k = sample_normal_below_n(len(x_old)-1)
                                if random_method == 'levy':
                                    k = sample_levy_below_n(len(x_old)-1)
                                if random_method == 'random':
                                    k = random.randint(0, len(x_old)-1)

                                x_,y_ = t_data(large_window, small_window, k, small_window)
                                x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=42)
                                x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)
                                criterion = nn.MSELoss()
                                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                                for epoch in range(num_epochs):
                                    model.train()
                                    outputs = model(x_train)
                                    loss = criterion(outputs, y_train)
                                    
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()

                        x , y = t_data(large_window, small_window, i, small_window)
                        x_old.append(i)

                        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                        x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        
                        for epoch in range(num_epochs):
                            model.train()
                            outputs = model(x_train)
                            loss = criterion(outputs, y_train)
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            y_pred_train = model(x_train)
                            y_pred_test = model(x_test)
                            train_loss = criterion(y_pred_train, y_train)
                            test_loss = criterion(y_pred_test, y_test)

                        y_pred_train = y_pred_train.cpu().detach().numpy().squeeze()
                        y_pred_test = y_pred_test.cpu().detach().numpy().squeeze()
                        y_train = y_train.cpu().detach().numpy().squeeze()
                        y_test = y_test.cpu().detach().numpy().squeeze()

                        predicted = np.vstack((y_pred_train, y_pred_test))
                        actual = np.vstack((y_train, y_test))

                        predicted_data[i+large_window,:] = predicted[-1,:]
                        actual_data[i+large_window,:] = actual[-1,:]



                    predicted_data = scaler.inverse_transform(predicted_data)
                    actual_data = scaler.inverse_transform(actual_data)
                    df_results = pd.DataFrame({
                        'Predicted_Threshold': predicted_data[:, 1],
                        'Best_Threshold': actual_data[:, 1]
                    })
                    csv_name= './result_gpu/sharpe/'+random_method+'/'+file_path
                    # Save to csv
                    df_results.to_csv(csv_name, index=False)