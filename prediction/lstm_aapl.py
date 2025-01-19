import pandas as pd
import torch
from torch import nn
from torch.nn import Transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings('ignore')
stocks_name = ['AAPL']
sentiment_type = ['nosentiment', 'noinfluence', 'withinfluence']
related_type = ['related', 'norelated', 'norelated_3']
window = [5, 15, 30]
his_long = 90
for window_size in window:
    for stock_name in stocks_name:
        for relate in related_type:
            for senti in sentiment_type:
                if relate == 'related':
                    csv_name = 'LSTM/LSTM_predict_'+stock_name+'_'+senti+str(window_size)+'.csv'
                else:
                    csv_name = 'LSTM/LSTM_predict_'+stock_name+'_'+senti+'_'+relate+str(window_size)+'.csv'


                data = pd.read_csv(stock_name + '_data_fill.csv')
                if senti == 'nosentiment':
                    if relate == 'norelated_3':
                        data = data[['Daily_Returns','Volatility', 'zeros']]
                        size = 3
                    elif relate == 'norelated':
                        data = data[['Daily_Returns','Volatility', 'zeros', 'zeros', 'zeros', 'zeros', 'zeros', 'zeros', 'zeros']]
                        size = 9
                    elif relate == 'related':
                        size = 9
                        if stock_name == 'AAPL':
                            data = data[['Daily_Returns','Volatility', 'MSFT_D','MSFT_V','GOOGL_D', 'GOOGL_V', 'AMZN_D', 'AMZN_V', 'zeros']]
                        if stock_name == 'JNJ':
                            data = data[['Daily_Returns','Volatility', 'ABT_D','ABT_V','MRK_D', 'MRK_V', 'AMGN_D', 'AMGN_V', 'zeros']]
                        if stock_name == 'JPM':
                            data = data[['Daily_Returns','Volatility', 'BAC_D','BAC_V','PNC_D', 'PNC_V', 'C_D', 'C_V', 'zeros']]
                        if stock_name == 'XOM':
                            data = data[['Daily_Returns','Volatility', 'CVX_D','CVX_V','COP_D', 'COP_V', 'HAL_D', 'HAL_V', 'zeros']]
                elif senti =='noinfluence':
                    if relate == 'norelated_3':
                        data = data[['Daily_Returns','Volatility', 'Sentiment no influence']]
                        size = 3
                    elif relate == 'norelated':
                        data = data[['Daily_Returns','Volatility', 'zeros', 'zeros', 'zeros', 'zeros', 'zeros', 'zeros', 'Sentiment no influence']]
                        size = 9
                    elif relate == 'related':
                        size = 9
                        if stock_name == 'AAPL':
                            data = data[['Daily_Returns','Volatility', 'MSFT_D','MSFT_V','GOOGL_D', 'GOOGL_V', 'AMZN_D', 'AMZN_V', 'Sentiment no influence']]
                        if stock_name == 'JNJ':
                            data = data[['Daily_Returns','Volatility', 'ABT_D','ABT_V','MRK_D', 'MRK_V', 'AMGN_D', 'AMGN_V', 'Sentiment no influence']]
                        if stock_name == 'JPM':
                            data = data[['Daily_Returns','Volatility', 'BAC_D','BAC_V','PNC_D', 'PNC_V', 'C_D', 'C_V', 'Sentiment no influence']]
                        if stock_name == 'XOM':
                            data = data[['Daily_Returns','Volatility', 'CVX_D','CVX_V','COP_D', 'COP_V', 'HAL_D', 'HAL_V', 'Sentiment no influence']]
                else:
                    if relate == 'norelated_3':
                        data = data[['Daily_Returns','Volatility', 'Sentiment with influence']]
                        size = 3
                    elif relate == 'norelated':
                        data = data[['Daily_Returns','Volatility', 'zeros', 'zeros', 'zeros', 'zeros', 'zeros', 'zeros', 'Sentiment with influence']]
                        size = 9
                    elif relate == 'related':
                        size = 9
                        if stock_name == 'AAPL':
                            data = data[['Daily_Returns','Volatility', 'MSFT_D','MSFT_V','GOOGL_D', 'GOOGL_V', 'AMZN_D', 'AMZN_V', 'Sentiment with influence']]
                        if stock_name == 'JNJ':
                            data = data[['Daily_Returns','Volatility', 'ABT_D','ABT_V','MRK_D', 'MRK_V', 'AMGN_D', 'AMGN_V', 'Sentiment with influence']]
                        if stock_name == 'JPM':
                            data = data[['Daily_Returns','Volatility', 'BAC_D','BAC_V','PNC_D', 'PNC_V', 'C_D', 'C_V', 'Sentiment with influence']]
                        if stock_name == 'XOM':
                            data = data[['Daily_Returns','Volatility', 'CVX_D','CVX_V','COP_D', 'COP_V', 'HAL_D', 'HAL_V', 'Sentiment with influence']]

                # 数据预处理
                scaler = MinMaxScaler()
                data = scaler.fit_transform(data)

                # 构建滑动窗口数据
                predicted_data = []
                actual_data = []
                for i in range(len(data) - window_size -his_long- 1): 
                    x = []
                    y = []

                    for j in range(his_long - window_size -1):
                        x.append(data[i+j:i+j+window_size, :])  # use all 9 columns as input
                        y.append(data[i+j+window_size, :size])  # we try to predict the first two columns in the future
                    x = np.array(x)
                    y = np.array(y)

                    #  Transform to tensors
                    x = torch.tensor(x, dtype=torch.float32)  # Shape: batch_size, sequence_length, num_features
                    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: batch_size, 1

                    # Split into training and test set
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

                    # 定义模型
                    class LSTMModel(nn.Module):
                        
                        def __init__(self, ninput, hidden_dim, noutput, num_layers=2):
                            super(LSTMModel, self).__init__()
                            
                            # LSTM layer
                            self.lstm = nn.LSTM(ninput, hidden_dim, num_layers, batch_first=True)
                            
                            # Fully connected layer
                            self.fc = nn.Linear(hidden_dim, noutput)
                        
                        def forward(self, x):
                            # LSTM expects input as (batch_size, sequence_length, num_features)
                            out, _ = self.lstm(x)
                            out = self.fc(out[:, -1, :])  # We only want the output of the last timestep
                            return out

                    # 参数设定
                    hidden_dim = 64

                    # 创建模型
                    model = LSTMModel(ninput=size, hidden_dim=hidden_dim, noutput=size)

                    # 定义损失函数和优化器
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                    # 训练模型
                    num_epochs = 100
                    for epoch in range(num_epochs):
                        model.train()
                        outputs = model(x_train)
                        loss = criterion(outputs, y_train)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        #if (epoch+1) % 10 == 0:
                            #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

                    # 模型评估
                    model.eval()
                    with torch.no_grad():
                        y_pred_train = model(x_train)
                        y_pred_test = model(x_test)
                        train_loss = criterion(y_pred_train, y_train)
                        test_loss = criterion(y_pred_test, y_test)
                        print(f'Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')

                    # Convert predictions and actual values to numpy arrays
                    y_pred_train = y_pred_train.detach().numpy().squeeze()
                    y_pred_test = y_pred_test.detach().numpy().squeeze()
                    y_train = y_train.detach().numpy().squeeze()
                    y_test = y_test.detach().numpy().squeeze()

                    # Stack them together
                    predicted = np.vstack((y_pred_train, y_pred_test))
                    actual = np.vstack((y_train, y_test))
                    predicted_data.append(predicted[-1])
                    actual_data.append(actual[-1])

                predicted_data = np.array(predicted_data)
                print(predicted_data.shape, stock_name, relate, senti, window_size)
                predicted_data = scaler.inverse_transform(predicted_data)
                actual_data = scaler.inverse_transform(actual_data)



                df_results = pd.DataFrame({
                    'Predicted_Daily_Returns': predicted_data[:, 0],
                    'Actual_Daily_Returns': actual_data[:, 0],
                    'Predicted_Volatility': predicted_data[:, 1],
                    'Actual_Volatility': actual_data[:, 1]
                })

                # Save to csv
                df_results.to_csv(csv_name, index=False)


