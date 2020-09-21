# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def prepare_vol(rolling_window=22, 
                prediction_window=22, 
                seq_len = 30,
                subsample=True,
                index='KOSPI2 Index'):
    price_data = pd.read_excel('price_data.xlsx', index_col=0).iloc[1:, :]
    price_data.index = pd.to_datetime(price_data.index)

    if subsample:
        price_data = price_data.loc['2001-01-01':'2017-01-01', :]


    KOSPI = price_data[[index]].dropna().copy()
    KOSPI['LOG_PRICE'] = np.log(KOSPI[index].astype(float))
    KOSPI['RETURN'] = KOSPI['LOG_PRICE'].diff()

    KOSPI = KOSPI.dropna()
    KOSPI_STD = KOSPI.rolling(window=rolling_window).std()

    data_set = pd.DataFrame({'VOL': KOSPI_STD['RETURN']})

    data_set['VOL+'+str(prediction_window)] = KOSPI_STD['RETURN'].shift(-prediction_window)

    data_set = data_set.dropna()
    return data_set


def prepare_explanary(subsample=True, index = 'KOSPI2 Index'):
    '''
    get explanary variables(all X except vol)
    '''    
    if index == 'SPX Index':
        country = 'us'
    elif index == 'KOSPI2 Index':
        country = 'korea'
    else:
        print('Wrong index!')
        exit()
    ytm_data = pd.read_excel('bond_ytm.xlsx', index_col=0).iloc[1:, :]
    ytm_data.index = pd.to_datetime(ytm_data.index)
    ytm_data = ytm_data[[country+' corp bond ytm', country+' gov bond ytm']]
    price_data = pd.read_excel('price_data.xlsx', index_col=0).iloc[1:, :]
    price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data[['XAU Curncy', 'USCRWTIC Index']]

    exp_data = pd.merge(ytm_data, price_data, left_index=True, right_index=True)

    if country == 'korea':
        file_name = 'kospi'
    elif country == 'us':
        file_name = 'spx'
    params_data = pd.read_csv(''+file_name+'_garch_variables_2000.csv', index_col=0)
    params_data.index = pd.to_datetime(params_data.index)
    exp_data = pd.merge(exp_data, params_data, left_index=True, right_index=True)

    if subsample:
        exp_data = exp_data.loc['2001-01-01':'2017-01-01', :]
    return exp_data

def prepare_data(exp_variables,
    rolling_window=22,
    prediction_window=22,
    seq_len = 30,
    subsample=False,
    split_point=0.66,
    index='KOSPI2 Index'):

    vol = prepare_vol(rolling_window, prediction_window, seq_len, subsample, index)
    exp_data = prepare_explanary(subsample, index)
    dataset = pd.merge(exp_data[exp_variables], vol, left_index=True, right_index=True, how='right')
    dataset = dataset.fillna(method='ffill')
    dataset = dataset.dropna()

    tmp_X = dataset.iloc[:, :-1]
    tmp_Y = dataset.iloc[:,[-1]]
    features = tmp_X.shape[1]
    li_X = []

    for i in range(seq_len):
        li_X.append(tmp_X.shift(i))
    li_X.append(tmp_Y)
    dataset_2 = pd.concat(li_X, axis=1)
    dataset_2 = dataset_2.dropna()

    X = dataset_2.iloc[:, :-1]
    Y = dataset_2.iloc[:,[-1]]
    li_X_2 = []
    for i in range(seq_len):
        tmp = torch.Tensor(X.iloc[:, features*i:features*(i+1)].values).view(1, -1, features)
        li_X_2.append(tmp)
    X = torch.cat(li_X_2, dim=0)
    Y = torch.Tensor(Y.values)
    Y = Y.T
    Y = Y.view((1, -1, 1))

    split = int(X.shape[1]*split_point)

    return {'train_set':[X[:, :split, :], Y[:, :split, :]], 
            'test_set':[X[:, split:, :], Y[:, split:, :]]}


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
class StandardScaler():
    """Standardize data by removing the mean and scaling to
    unit variance.  This object can be used as a transforma
    in PyTorch data loaders.

    Args:
        mean (FloatTensor): The mean value for each feature in the data.
        scale (FloatTensor): Per-feature relative scaling.
    """

    def __init__(self, mean=None, scale=None):
        if mean is not None:
            mean = torch.FloatTensor(mean)
        if scale is not None:
            scale = torch.FloatTensor(scale)
        self.mean_ = mean
        self.scale_ = scale

    def fit(self, sample):
        """Set the mean and scale values based on the sample data.
        """
        self.mean_ = sample.mean(1, keepdim=True).to(device)
        self.scale_ = sample.std(1, unbiased=False, keepdim=True).to(device)
        return self

    def transform(self, sample):
        return (sample - self.mean_)/self.scale_

    def inverse_transform(self, sample):
        """Scale the data back to the original representation
        """
        return sample * self.scale_ + self.mean_

torch.manual_seed(1)


class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,layer_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.dropout = nn.Dropout(p=0.3)
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, dropout=0.8)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), 1)
    
    def forward(self,voltility):
        voltility = self.dropout(voltility)
        lstm_out, _ = self.lstm(voltility)
        next_vol = self.fc1(lstm_out[[-1], :, :])
        next_vol = self.fc2(next_vol)
        return next_vol


def split_test(input_size,hidden_size,layer_size, exp_variables, index, split=0.628515):
    global plot
    model = LSTM(input_size,hidden_size,layer_size).to(device)
    loss_function = F.torch.nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)

    data_set = prepare_data(exp_variables, index=index, split_point = split)
    train_X, train_Y = data_set['train_set']
    test_X, test_Y = data_set['test_set']
    
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    train_X_scaler =StandardScaler().fit(train_X)
    train_X = train_X_scaler.transform(train_X)
    train_Y_scaler = StandardScaler().fit(train_Y)
    train_Y = train_Y_scaler.transform(train_Y)

    test_X = train_X_scaler.transform(test_X)
    test_Y = train_Y_scaler.transform(test_Y)


    
    train_loss = []
    test_loss = []
    for epoch in range(2000):
        model.zero_grad()
        output = model(train_X)
        loss = loss_function(output, train_Y)
        loss.backward()
        opt.step()
        train_loss.append(loss.item())
        test_loss.append(loss_function(model(test_X), test_Y).item())

        if epoch%50 == 0 and plot:
            print("epoch: ", epoch, ', loss: ', loss.item())

    with torch.no_grad():
        OOS_Y = model(test_X).detach()
        output = output.detach()
        OOS_Y = train_Y_scaler.inverse_transform(OOS_Y).cpu().numpy().flat
        Test_Y = train_Y_scaler.inverse_transform(test_Y).cpu().numpy().flat
        
        Train_Y = train_Y_scaler.inverse_transform(train_Y).cpu().numpy().flat
        Train_ouput_Y = train_Y_scaler.inverse_transform(output).cpu().numpy().flat
        
        test_merged = pd.DataFrame({'RV':Test_Y, 'pre':OOS_Y})
        train_merged = pd.DataFrame({'RV':Train_Y, 'pre':Train_ouput_Y})
    
        test_MAE = np.abs(test_merged.iloc[:,0] - test_merged.iloc[:,1]).mean()
        train_MAE = np.abs(train_merged.iloc[:,0] - train_merged.iloc[:,1]).mean()
        total_MAE = (test_MAE * len(test_merged) + train_MAE * len(train_merged)) / (len(test_merged) + len(train_merged))
        MAE = [train_MAE*100, test_MAE*100, total_MAE*100]

        test_MSE = np.square(test_merged.iloc[:,0] - test_merged.iloc[:,1]).mean()
        train_MSE = np.square(train_merged.iloc[:,0] - train_merged.iloc[:,1]).mean()
        total_MSE = (test_MSE * len(test_merged) + train_MSE * len(train_merged)) / (len(test_merged) + len(train_merged))
        MSE = [train_MSE*10000, test_MSE*10000, total_MSE*10000]
        
        test_HMAE = np.abs(1 - test_merged.iloc[:,1] / test_merged.iloc[:,0]).mean()
        train_HMAE = np.abs(1 - train_merged.iloc[:,1] / train_merged.iloc[:,0]).mean()
        total_HMAE = (test_HMAE * len(test_merged) + train_HMAE * len(train_merged)) / (len(test_merged) + len(train_merged))
        HMAE = [train_HMAE, test_HMAE, total_HMAE]
    
        test_HMSE = np.square(1 - test_merged.iloc[:,1] / test_merged.iloc[:,0]).mean()
        train_HMSE = np.square(1 - train_merged.iloc[:,1] / train_merged.iloc[:,0]).mean()
        total_HMSE = (test_HMSE * len(test_merged) + train_HMSE * len(train_merged)) / (len(test_merged) + len(train_merged))
        HMSE = [train_HMSE, test_HMSE, total_HMSE]
        ret = pd.DataFrame({'MAE':MAE, 'MSE':MSE, 'HMAE':HMAE, 'HMSE':HMSE}, index=['Train', 'Test', 'Total'])
        
        fig, ax = plt.subplots(2, 2, figsize=(18, 12))
        ax[0][0].plot(train_Y_scaler.inverse_transform(train_Y).cpu().numpy().flat)
        ax[0][0].plot(train_Y_scaler.inverse_transform(model(train_X)).detach().cpu().numpy().flat)
        ax[0][0].legend(['true', 'predict'])
        ax[0][0].set_title('Train Set')

        ax[0][1].plot(Test_Y)
        ax[0][1].plot(OOS_Y)
        ax[0][1].legend(['true', 'predict'])
        ax[0][1].set_title('Test Set')

        ax[1][0].plot(train_loss)
        ax[1][0].set_title('Train Loss')

        ax[1][1].plot(test_loss)
        ax[1][1].set_title('Test Loss')

        plt.show()
        print(ret)
        
def rolling_test(input_size,hidden_size,layer_size, exp_variables, index, split=0.628515):
    global plot


    data_set = prepare_data(exp_variables, index=index, split_point = split)
    train_X, train_Y = data_set['train_set']
    test_X, test_Y = data_set['test_set']
    
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    train_X_scaler =StandardScaler().fit(train_X)
    train_X = train_X_scaler.transform(train_X)
    train_Y_scaler = StandardScaler().fit(train_Y)
    train_Y = train_Y_scaler.transform(train_Y)

    test_X = train_X_scaler.transform(test_X)
    test_Y = train_Y_scaler.transform(test_Y)


    
    train_loss = []
    test_loss = []
    Test_Y = []
    OOS_Y = []
    Train_Y = []
    Train_output_Y = []
    for i in range(int(test_Y.shape[1]/22)):
        model = LSTM(input_size,hidden_size,layer_size).to(device)
        loss_function = F.torch.nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=0.01)
        model.train()
        this_train_X = torch.cat([train_X[:, i*22:, :], test_X[:, :i*22, :]], axis=1)
        this_train_Y = torch.cat([train_Y[:, i*22:, :], test_Y[:, :i*22, :]], axis=1)
        this_test_X = test_X[:, i*22:(i+1)*22, :]
        this_test_Y = test_Y[:, i*22:(i+1)*22, :]
        
        for epoch in range(1000):
            model.zero_grad()
            output = model(this_train_X)
            loss = loss_function(output, this_train_Y)
            loss.backward()
            opt.step()
                
        model.eval()
        
        with torch.no_grad():
            output = output.detach()
            OOS_Y.extend(train_Y_scaler.inverse_transform(model(this_test_X).detach()).cpu().numpy().reshape(-1).tolist())
            Test_Y.extend(train_Y_scaler.inverse_transform(this_test_Y).cpu().numpy().reshape(-1).tolist())

            Train_Y.extend(train_Y_scaler.inverse_transform(this_train_Y).cpu().numpy().reshape(-1).tolist())
            Train_output_Y.extend(train_Y_scaler.inverse_transform(output).cpu().numpy().reshape(-1).tolist())
        
    test_merged = pd.DataFrame({'RV':Test_Y, 'pre':OOS_Y})
    train_merged = pd.DataFrame({'RV':Train_Y, 'pre':Train_output_Y})
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_merged)
    ax.legend(['Realized Vol', 'Predicted Vol'])
    test_MAE = np.abs(test_merged.iloc[:,0] - test_merged.iloc[:,1]).mean()
    train_MAE = np.abs(train_merged.iloc[:,0] - train_merged.iloc[:,1]).mean()
    total_MAE = (test_MAE * len(test_merged) + train_MAE * len(train_merged)) / (len(test_merged) + len(train_merged))
    MAE = [train_MAE*100, test_MAE*100, total_MAE*100]

    test_MSE = np.square(test_merged.iloc[:,0] - test_merged.iloc[:,1]).mean()
    train_MSE = np.square(train_merged.iloc[:,0] - train_merged.iloc[:,1]).mean()
    total_MSE = (test_MSE * len(test_merged) + train_MSE * len(train_merged)) / (len(test_merged) + len(train_merged))
    MSE = [train_MSE*10000, test_MSE*10000, total_MSE*10000]

    test_HMAE = np.abs(1 - test_merged.iloc[:,1] / test_merged.iloc[:,0]).mean()
    train_HMAE = np.abs(1 - train_merged.iloc[:,1] / train_merged.iloc[:,0]).mean()
    total_HMAE = (test_HMAE * len(test_merged) + train_HMAE * len(train_merged)) / (len(test_merged) + len(train_merged))
    HMAE = [train_HMAE, test_HMAE, total_HMAE]

    test_HMSE = np.square(1 - test_merged.iloc[:,1] / test_merged.iloc[:,0]).mean()
    train_HMSE = np.square(1 - train_merged.iloc[:,1] / train_merged.iloc[:,0]).mean()
    total_HMSE = (test_HMSE * len(test_merged) + train_HMSE * len(train_merged)) / (len(test_merged) + len(train_merged))
    HMSE = [train_HMSE, test_HMSE, total_HMSE]
    ret = pd.DataFrame({'MAE':MAE, 'MSE':MSE, 'HMAE':HMAE, 'HMSE':HMSE}, index=['Train', 'Test', 'Total'])

    print(ret)

# Experiments Starts
# %%
exp_variables = []
plot = False
INPUT_SIZE = len(exp_variables) + 1
HIDDEN_SIZE = 16
print('Split Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    split_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')
print('Rolling Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    rolling_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')


# %%
exp_variables = ['us corp bond ytm', 'us gov bond ytm', 'XAU Curncy', 'USCRWTIC Index', ]
plot = False
INPUT_SIZE = len(exp_variables) + 1
HIDDEN_SIZE = 16
print('Split Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    split_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')
print('Rolling Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    rolling_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')
    


# %%
exp_variables = ['garch_cond_vol', 'garch_resid']
plot = False
INPUT_SIZE = len(exp_variables) + 1
HIDDEN_SIZE = 16
print('Split Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    split_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')
print('Rolling Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    rolling_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')


# %%
exp_variables = [ 'egarch_cond_vol','egarch_std_resid', 'egarch_asymmetric']
plot = False
INPUT_SIZE = len(exp_variables) + 1
HIDDEN_SIZE = 16
print('Split Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    split_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')
print('Rolling Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    rolling_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')


# %%
exp_variables = ['ewma_cond_vol', 'ewma_resid']
plot = False
INPUT_SIZE = len(exp_variables) + 1
HIDDEN_SIZE = 16
print('Split Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    split_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')
print('Rolling Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    rolling_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')


# %%
exp_variables = ['garch_cond_vol', 'garch_resid', 'egarch_cond_vol','egarch_std_resid', 'egarch_asymmetric']
plot = False
INPUT_SIZE = len(exp_variables) + 1
HIDDEN_SIZE = 16
print('Split Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    split_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')
print('Rolling Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    rolling_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')


# %%
exp_variables = ['egarch_cond_vol','egarch_std_resid', 'egarch_asymmetric', 'ewma_cond_vol', 'ewma_resid']
plot = False
INPUT_SIZE = len(exp_variables) + 1
HIDDEN_SIZE = 16
print('Split Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    split_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')
print('Rolling Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    rolling_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')


# %%
exp_variables = ['garch_cond_vol', 'garch_resid', 'ewma_cond_vol', 'ewma_resid']
plot = False
INPUT_SIZE = len(exp_variables) + 1
HIDDEN_SIZE = 16
print('Split Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    split_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')
print('Rolling Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    rolling_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')


# %%
exp_variables = ['garch_cond_vol', 'garch_resid', 'egarch_cond_vol','egarch_std_resid', 'egarch_asymmetric', 'ewma_cond_vol', 'ewma_resid']
plot = False
INPUT_SIZE = len(exp_variables) + 1
HIDDEN_SIZE = 16

print('Split Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    split_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')
print('Rolling Test')
for LAYER_SIZE in range(1, 4):
    print('layer = ', LAYER_SIZE)
    rolling_test(INPUT_SIZE, HIDDEN_SIZE, LAYER_SIZE, exp_variables, index='SPX Index')


