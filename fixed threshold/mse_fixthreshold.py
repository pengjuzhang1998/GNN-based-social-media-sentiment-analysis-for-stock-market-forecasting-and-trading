import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
import warnings

warnings.filterwarnings('ignore')
stocks_name = ['AAPL', 'JNJ', 'JPM', 'XOM']
sentiment_type = ['nosentiment', 'noinfluence', 'withinfluence']
related_type = ['related', 'norelated', 'norelated_3']
window = [5, 15, 30]
analyse_data = pd.DataFrame()
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

data_d = np.zeros(1000)
data_s = np.zeros(1000)
t_cost = 0.002
for stock_name in stocks_name:
    for window_size in window:
        for relate in related_type:
            for senti in sentiment_type:
                if relate == 'related':
                    csv_name = 'CNN/CNN_predict_'+stock_name+'_'+senti+str(window_size)+'.csv'
                else:
                    csv_name = 'CNN/CNN_predict_'+stock_name+'_'+senti+'_'+relate+str(window_size)+'.csv'

                df = pd.read_csv(csv_name)

                p_d = df['Predicted_Daily_Returns']
                p_v = df['Predicted_Volatility']
                a_d = df['Actual_Daily_Returns']
                a_v = df['Actual_Volatility']
                def normalize_data(data):
                    # 获取小于0的部分的最小值
                    min_val = np.min(data[data < 0]) if len(data[data < 0]) > 0 else 0
                    
                    # 获取大于0的部分的最大值
                    max_val = np.max(data[data > 0]) if len(data[data > 0]) > 0 else 1
                    
                    normalized_data = []
                    for value in data:
                        if value < 0:
                            normalized_value = 0.5 * ((value - min_val) / (0 - min_val))
                        else:
                            normalized_value = 0.5 + 0.5 * (value / max_val)
                        normalized_data.append(normalized_value)
                    
                    return np.array(normalized_data)
                max_P = abs(df['Predicted_Daily_Returns'].max())
                result_d = []
                result_s = []
                T0 = 0
                R0 = 0
                T1 = 0
                R1 = 0
                p_d = df['Predicted_Daily_Returns']
                p_v = df['Predicted_Volatility']
                a_d = df['Actual_Daily_Returns']
                a_v = np.array(df['Actual_Volatility'])
                a_d = np.array(a_d)
                best_data_d = np.zeros(len(p_d))
                best_data_s = np.zeros(len(p_d))
                d_T = []
                s_T = []
                for T in np.linspace(0, max_P, 1000):
                    fund_d = Funding(False, 100)
                    temp_data = np.zeros(len(p_d))
                    for i in range(len(p_d)):
                        trading(p_dailyreturn = p_d[i], a_dailyreturn = a_d[i], funding= fund_d, T_posi=T, T_nega=-T, t_cost = t_cost)
                        temp_data[i] = fund_d.price
                    if fund_d.price>R0:
                        R0 = fund_d.price
                        T0 = T
                        best_data_d = temp_data
                    result_d.append(fund_d.price)
                    d_T.append(T)
                for T in np.linspace(0, max_P/a_v.mean(), 1000):
                    fund_s = Funding(False, 100)
                    for i in range(len(p_d)):
                        trading(p_dailyreturn = p_d[i]/a_v[i], a_dailyreturn = a_d[i], funding= fund_s, T_posi=T, T_nega=-T, t_cost = t_cost)
                        temp_data[i] = fund_s.price
                    if fund_s.price>R1:
                        R1 = fund_s.price
                        T1 = T
                        best_data_s = temp_data
                    result_s.append(fund_s.price)
                    s_T.append(T)
                mse_r = np.mean((p_d - a_d) ** 2)/np.mean(a_d)

                correct_rate = 0
                for i in range(len(p_d)):
                    if p_d[i]*a_d[i]>=0:
                        correct_rate = correct_rate+1

                correct_rate = correct_rate/len(p_d)
                a_d[a_d < 0] = 0
                a_d[a_d > 0] = 1
                #p_d = normalize_data(p_d.values)
                #b_score_r = brier_score_loss(a_d, p_d)
                #analyse_data[csv_name] = [mse_r, correct_rate, b_score_r,R0, T0, R1, T1]
                print(mse_r, correct_rate,R0, T0, R1, T1)
                #best_df_d = pd.DataFrame(best_data_d)
                threshold_d = pd.DataFrame(result_d, d_T)
                threshold_s = pd.DataFrame(result_s, s_T)
                threshold_d.to_csv(csv_name[:-4]+'thresholddaily.csv', index=False)
                threshold_s.to_csv(csv_name[:-4]+'thresholdsharpe.csv', index=False)
                data_d = data_d + np.array(result_d)
                data_s = data_s + np.array(result_s)
                #best_df_d.to_csv(csv_name[:-4]+'daily.csv', index=False)
                #best_df_s = pd.DataFrame(best_data_s)
                #best_df_s.to_csv(csv_name[:-4]+'sharpe.csv', index=False)
data_d = data_d/108
data_s = data_s/108
ave_trading = pd.DataFrame([data_d, data_s])
ave_trading.to_csv('avetrading_CNN_fix.csv', index=False)
#analyse_data.to_csv('analyse_transformer_remake.csv', index=False)
