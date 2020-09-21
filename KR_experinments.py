#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


# In[16]:


pd.__version__


# In[17]:


device = torch.device('cuda:0')


# In[18]:


df1 = pd.read_excel('price_data.xlsx')


df1.drop(index=0,inplace=True)
df1.reset_index(drop=True,inplace=True)
# print(df.columns)
# print(df['Unnamed: 0'])

from datetime import datetime


df1['datetime'] = pd.to_datetime(df1['Unnamed: 0'], format = '%Y-%m-%d %H:%M:%S')
df1 = df1.set_index(['datetime'])
df2 = pd.read_csv('kospi_garch_variables_2000.csv')
df2['datetime'] = pd.to_datetime(df2['date'], format = '%Y-%m-%d')
df2 = df2.set_index(['datetime'])

df = pd.concat([df1, df2], axis=1, join='inner')


variable_list = ['KOSPI2 Index', 'garch_cond_vol', 'garch_resid', 'egarch_cond_vol','egarch_std_resid', 'egarch_asymmetric', 'ewma_cond_vol', 'ewma_resid']

training_set = df[variable_list].copy()

training_set = training_set.dropna(how='any')

log_return = np.array([np.log(123.86)-np.log(133.66)])

b = np.log(np.array(training_set['KOSPI2 Index'][:-1].values).astype('float')) - np.log(np.array(training_set['KOSPI2 Index'][1:].values).astype('float'))
training_set['log_return'] = np.append(log_return,b)
vol = training_set['log_return']
roller = vol.rolling(22)
volList = roller.std(ddof=0)
training_set['Volatility'] = pd.DataFrame(volList)
training_set = training_set.dropna(how='any')


train_size = 3316
test_size = len(training_set) - train_size


sc = MinMaxScaler()
variable_list = ['KOSPI2 Index', 'garch_cond_vol', 'garch_resid', 'egarch_cond_vol','egarch_std_resid', 'egarch_asymmetric',
                 'ewma_cond_vol', 'ewma_resid','log_return','Volatility']
variable_list_x = ['KOSPI2 Index', 'garch_cond_vol', 'garch_resid', 'egarch_cond_vol','egarch_std_resid', 'egarch_asymmetric',
                 'ewma_cond_vol', 'ewma_resid','log_return']
for var in variable_list_x:
    training_set[var] /= max(training_set[var])-min(training_set[var])


# In[19]:


def sliding_windows(data,var_list, seq_length):
    x = []
    y = []

    for i in range(len(data['Volatility'])-2*seq_length-1):

        _x = np.array(data[var_list][i:(i+seq_length)])
        _y = data['Volatility'][i+2*seq_length]

        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

# num is the num of model, when num = 0, it is our first model.
def data_prepare(data,var_list, seq_length,train_size=3315):
    x, y = sliding_windows(training_set,var_list, seq_length)

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    y = np.array([y]).T
    dataX = Variable(torch.Tensor(x))
    dataY = Variable(torch.Tensor(y))
    # print(dataY)
    trainX = Variable(torch.Tensor(x[:train_size]))
    trainY = Variable(torch.Tensor(y[:train_size]))

    testX = Variable(torch.Tensor(x[train_size:]))
    testY = Variable(torch.Tensor(y[train_size:]))
    
    return dataX.to(device),dataY.to(device),trainX.to(device),trainY.to(device),testX.to(device),testY.to(device)
## l1loss, batch size = whole data
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 22
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out
    
def Error_stat(dataY_plot,data_predict,train_size):
    MAE_test = sum(abs(dataY_plot[train_size:] - data_predict[train_size:]))/(len(dataY_plot)-train_size)
    MSE_test = sum((dataY_plot[train_size:] - data_predict[train_size:])**2)/(len(dataY_plot)-train_size)
    HMAE_test = sum(abs(1-dataY_plot[train_size:]/data_predict[train_size:]))/(len(dataY_plot)-train_size)
    HMSE_test = sum((1-dataY_plot[train_size:]/data_predict[train_size:])**2)/(len(dataY_plot)-train_size)
    MAE_train = sum(abs(dataY_plot[:train_size] - data_predict[:train_size]))/train_size
    MSE_train = sum((dataY_plot[:train_size] - data_predict[:train_size])**2)/train_size
    HMAE_train = sum(abs(1-dataY_plot[:train_size]/data_predict[:train_size]))/train_size
    HMSE_train = sum((1-dataY_plot[:train_size]/data_predict[:train_size])**2)/train_size
    MAE_overall = sum(abs(dataY_plot - data_predict))/len(dataY_plot)
    MSE_overall = sum((dataY_plot - data_predict)**2)/len(dataY_plot)
    HMAE_overall = sum(abs(1-dataY_plot/data_predict))/len(dataY_plot)
    HMSE_overall = sum((1-dataY_plot/data_predict)**2)/len(dataY_plot)
    final_stat = pd.DataFrame([['','Test','Train','Overall'],
                               ['MAE',MAE_test*100,MAE_train*100,MAE_overall*100],
                               ['MSE',MSE_test*10000,MSE_train*10000,MSE_overall*10000],
                               ['HMAE',HMAE_test,HMAE_train,HMAE_overall],
                               ['HMSE',HMSE_test,HMSE_train,HMSE_overall]])
    return final_stat

def all_together(var_list):
    seq_length = 22

    
    dataX,dataY,trainX,trainY,testX,testY = data_prepare(training_set,var_list, seq_length)

    num_epochs = 2000
    learning_rate = 0.001
    input_size = len(var_list)
    hidden_size = 22*input_size
    num_layers = 1
    num_classes = 1

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)
    criterion = torch.nn.L1Loss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    loss_list = []


    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(outputs, trainY)
        loss_list.append(loss.item())
        loss.backward()

        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    plt.plot(loss_list)
    plt.suptitle('Loss')
    plt.show()

    lstm.eval()

            
    train_predict = lstm(dataX)

    data_predict = train_predict.data.cpu().numpy()

    dataY_plot = dataY.data.cpu().numpy()

    plt.figure(figsize=(20,10))
    plt.axvline(x=train_size, c='r', linestyle='--')

    table = Error_stat(dataY_plot,data_predict,3316)
    display(table)
    plt.plot(dataY_plot)
    plt.plot(data_predict)
    plt.suptitle('Time-Series Prediction')
    plt.show()


# ## 3-Garch-Type-Parameter

# In[20]:


var_list = ['Volatility','KOSPI2 Index', 'garch_cond_vol','garch_resid', 'egarch_cond_vol', 'egarch_std_resid', 
            'egarch_asymmetric','ewma_cond_vol', 'ewma_resid']
all_together(var_list)


# ## Garch+EGarch

# In[21]:


var_list = ['Volatility','KOSPI2 Index', 'garch_cond_vol','garch_resid', 'egarch_cond_vol', 'egarch_std_resid']
all_together(var_list)


# ## Garch + EWMA

# In[22]:


var_list = ['Volatility','KOSPI2 Index', 'garch_cond_vol','garch_resid', 'ewma_cond_vol', 'ewma_resid']
all_together(var_list)


# ## EWMA+EGARCH

# In[23]:


var_list = ['Volatility','KOSPI2 Index', 'egarch_cond_vol', 'egarch_std_resid', 
            'egarch_asymmetric','ewma_cond_vol', 'ewma_resid']
all_together(var_list)


# ## Garch

# In[24]:


var_list = ['Volatility','KOSPI2 Index', 'garch_cond_vol','garch_resid']
all_together(var_list)


# ## EGARCH

# In[25]:


var_list = ['Volatility','KOSPI2 Index', 'egarch_cond_vol', 'egarch_std_resid', 'egarch_asymmetric']
all_together(var_list)


# ## EWMA

# In[26]:


var_list = ['Volatility','KOSPI2 Index','ewma_cond_vol', 'ewma_resid']
all_together(var_list)


# ## Single Model

# In[27]:


var_list = ['Volatility','KOSPI2 Index']
all_together(var_list)


# In[28]:


table = Error_stat(dataY_plot,data_predict,train_size)
table


# In[ ]:





# In[ ]:




