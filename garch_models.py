# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from IPython.display import display, Markdown
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.arima_model import ARMA
from arch.univariate import ConstantMean, LS, ARX, GARCH, EGARCH, EWMAVariance
import arch
from math import sqrt, log, pi, pow

import datetime as dt
import matplotlib as mpl

plt.style.use('seaborn')    
pd.set_option('precision', 5)

from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
years     = YearLocator(5) 
months    = MonthLocator() 
years_fmt = DateFormatter('%Y')

mpl.rcParams['figure.figsize'] =(30, 8)  
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16

# %% [markdown]
# ## Load Data

# %%
data = pd.read_excel('price_data.xlsx')


# %%
data2 = pd.read_excel('1990_2000_price_data.xlsx')


# %%
data = data.iloc[1:, :]
data.index = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d')
data.index.name = 'date'
data = data.iloc[:, 1:].astype('float')


# %%
data2 = data2.iloc[1:, :]
data2.index = pd.to_datetime(data2.iloc[:, 0], format='%Y-%m-%d')
data2.index.name = 'date'
data2 = data2.iloc[:, 1:].astype('float')
data2.dropna(inplace=True)


# %%
kospi_data = pd.concat([data2['KOSPI2 Index'], data['KOSPI2 Index']])
spx_data = pd.concat([data2['SPX Index'], data['SPX Index']])                     


# %%
kospi_data = kospi_data.sort_index().dropna()
spx_data = spx_data.sort_index().dropna()


# %%
kospi_data.plot()
spx_data.plot()


# %%
kospi_ret_data = np.log(kospi_data).diff().dropna()
spx_ret_data = np.log(spx_data).diff().dropna()
ret_data = np.log(data).diff().dropna()


# %%
kospi_ret_data.plot()
spx_ret_data.plot()

# %% [markdown]
# ## Single GARCH Model Result - Entire Dataset

# %%
start_date = dt.datetime(2000, 1, 3)
split_date = dt.datetime(2013, 12, 31)

y1 = spx_ret_data.loc[spx_ret_data.index > start_date].to_frame()
y2 = kospi_ret_data.loc[kospi_ret_data.index > start_date].to_frame()


# %%
def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = smt.stattools.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# %%
adf_test(y1)


# %%
adf_test(y2)


# %%
rvol1 = y1.rolling(window=22).std().shift(-21).dropna()
rvol2 = y2.rolling(window=22).std().shift(-21).dropna()


# %%
def run_garch(y, rvol, model, split_date, x=None, verbose=True, lam=None):

    # specify mean model
    ls = ConstantMean(y=y)
    
    # specify volatility model
    if model == "GARCH":
        ls.volatility = GARCH(p=1, q=1)
    elif model == "EGARCH":
        ls.volatility = EGARCH(p=1, o=1, q=1)
    elif model == "EWMA":
        ls.volatility = EWMAVariance(lam)
    else:
        print("Misspecified volatility process name")
    
    res = ls.fit(disp='off', last_obs=split_date)
    
    forecasts_1d = res.forecast(horizon=1)
    forecasted_vol = forecasts_1d.variance.pow(0.5).shift(1).dropna()
    
    test_merged = rvol.join(forecasted_vol).dropna()
    train_merged = rvol.join(res.conditional_volatility).dropna()

    test_MAE = np.abs(test_merged.iloc[:,0] - test_merged.iloc[:,1]).mean()
    train_MAE = np.abs(train_merged.iloc[:,0] - train_merged.iloc[:,1]).mean()
    total_MAE = (test_MAE * len(test_merged) + train_MAE * len(train_merged)) / (len(test_merged) + len(train_merged))
    MAE = [train_MAE, test_MAE, total_MAE]
    
    test_MSE = np.square(test_merged.iloc[:,0] - test_merged.iloc[:,1]).mean()
    train_MSE = np.square(train_merged.iloc[:,0] - train_merged.iloc[:,1]).mean()
    total_MSE = (test_MSE * len(test_merged) + train_MSE * len(train_merged)) / (len(test_merged) + len(train_merged))
    MSE = [train_MSE, test_MSE, total_MSE]
    
    test_HMAE = np.abs(1 - test_merged.iloc[:,1] / test_merged.iloc[:,0]).mean()
    train_HMAE = np.abs(1 - train_merged.iloc[:,1] / train_merged.iloc[:,0]).mean()
    total_HMAE = (test_HMAE * len(test_merged) + train_HMAE * len(train_merged)) / (len(test_merged) + len(train_merged))
    HMAE = [train_HMAE, test_HMAE, total_HMAE]
    
    test_HMSE = np.square(1 - test_merged.iloc[:,1] / test_merged.iloc[:,0]).mean()
    train_HMSE = np.square(1 - train_merged.iloc[:,1] / train_merged.iloc[:,0]).mean()
    total_HMSE = (test_HMSE * len(test_merged) + train_HMSE * len(train_merged)) / (len(test_merged) + len(train_merged))
    HMSE = [train_HMSE, test_HMSE, total_HMSE]

    df_results = pd.DataFrame(data=np.c_[MAE, MSE, HMAE, HMSE].T, columns=[model + ' ' + x for x in ['in-sample', 'out-of-sample', 'total']],                               index=['MAE', 'MSE', 'HMAE', 'HMSE']).T

    if verbose:
        
        display(Markdown('#### <br> <br> GARCH model results'))
        print(res.summary())
        
        display(Markdown('#### <br> <br> Plot forecast by model vs realized vol'))
        ax = plt.gca()
        forecasted_vol.plot(color='g', ax=ax, alpha=1, label='prediction oos')
        rvol.plot(color='blue', ax=ax, label='ground truth')
        res.conditional_volatility.plot(color='orange', ax=ax, label='prediction in-sample')
        ax.legend()
        
        display(Markdown('#### <br> <br> Results of out-of-sample forecasts with various loss functions'))
        display(df_results)
        
    return df_results


# %%
garch_results1 = run_garch(y=y1*100, rvol=rvol1*100, model='GARCH', split_date=split_date, verbose=False)
garch_results2 = run_garch(y=y2*100, rvol=rvol2*100, model='GARCH', split_date=split_date, verbose=False)
egarch_results1 = run_garch(y=y1*100, rvol=rvol1*100, model='EGARCH', split_date=split_date, verbose=False)
egarch_results2 = run_garch(y=y2*100, rvol=rvol2*100, model='EGARCH', split_date=split_date, verbose=False)
ewma_results1 = run_garch(y=y1*100, rvol=rvol1*100, model='EWMA', split_date=split_date, verbose=False)
ewma_results2 = run_garch(y=y2*100, rvol=rvol2*100, model='EWMA', split_date=split_date, verbose=False)


# %%
us_result = pd.concat([garch_results1, egarch_results1, ewma_results1])
us_result


# %%
kr_result = pd.concat([garch_results2, egarch_results2, ewma_results2])
kr_result

# %% [markdown]
# ## Prepare GARCH variables for LSTM training - Entire Dataset

# %%
def run_garch_simple(y, mean_model, vol_model, split_date, x=None, verbose=False):

    # specify mean model
    if mean_model == "CONST":
        ls = ConstantMean(y)
    elif mean_model == 'LS':
        ls = LS(y=y, x=x)
    elif mean_model == 'ARX':
        ls = ARX(y=y, lags=1)
    else:
        print("Misspecified mean model name. Please choose between CONST, LS, ARX.")
    
    # specify volatility model
    if vol_model == "GARCH":
        ls.volatility = GARCH(p=1, q=1)
    elif vol_model == "EGARCH":
        ls.volatility = EGARCH(p=1, o=1, q=1)
    elif vol_model == "EWMA":
        ls.volatility = EWMAVariance(lam=None)
    else:
        print("Misspecified volatility process name. Please choose between GARCH, EGARCH, EWMA.")
    
    res = ls.fit(disp='off', last_obs=split_date)
    
    if verbose:
        display(Markdown('#### <br> <br> GARCH model results'))
        print(res.summary())
    
    return res


# %%
def generate_garch_variables(ret_data, split_date):
    
    # GARCH
    garch_res = run_garch_simple(ret_data * 100, mean_model='CONST', vol_model='GARCH', split_date=split_date)
    garch_forecast = garch_res.forecast(horizon=1)
    
    garch_cond_vol_train= garch_res.conditional_volatility.shift(1).dropna().to_frame()
    garch_cond_vol_test = np.sqrt(garch_forecast.variance.shift(2).dropna())
    garch_cond_vol_train.columns = ['cond_vol']
    garch_cond_vol_test.columns = ['cond_vol']
    garch_cond_vol = pd.concat([garch_cond_vol_train, garch_cond_vol_test], axis=0)
    garch_resid = ret_data.shift(1).dropna() - garch_res.params[0]
    
    garch_var1 = garch_res.params[3] * np.square(garch_cond_vol)
    garch_var1.columns = ['garch_cond_vol']
    garch_var2 = garch_res.params[2] * np.square(garch_resid).to_frame()
    garch_var2.columns = ['garch_resid']

    # EGARCH
    egarch_res = run_garch_simple(ret_data * 100, mean_model='CONST', vol_model='EGARCH', split_date=split_date)
    egarch_forecast = egarch_res.forecast(horizon=1)
    
    egarch_cond_vol_train = egarch_res.conditional_volatility.shift(1).dropna().to_frame()
    egarch_cond_vol_test = np.sqrt(egarch_forecast.variance.shift(2).dropna())
    egarch_cond_vol_train.columns = ['cond_vol']
    egarch_cond_vol_test.columns = ['cond_vol']
    egarch_cond_vol = pd.concat([egarch_cond_vol_train, egarch_cond_vol_test], axis=0)
    egarch_resid = ret_data.shift(1).dropna() - egarch_res.params[0]
    egarch_std_resid = egarch_resid / egarch_cond_vol.iloc[:, 0]
    
    egarch_var1 = egarch_res.params[4] * np.log(np.square(egarch_cond_vol))
    egarch_var1.columns = ['egarch_cond_vol']
    egarch_var2 = egarch_res.params[3] * egarch_std_resid.to_frame()
    egarch_var2.columns = ['egarch_std_resid']
    egarch_var3 = egarch_res.params[2] * (np.abs(egarch_std_resid) - np.sqrt(2 / np.pi)).to_frame()
    egarch_var3.columns = ['egarch_asymmetric']

    # EWMA
    ewma_res = run_garch_simple(ret_data * 100, mean_model='CONST', vol_model='EWMA', split_date=split_date)
    ewma_forecast = ewma_res.forecast(horizon=1)
    
    ewma_cond_vol_train = ewma_res.conditional_volatility.shift(1).dropna().to_frame()
    ewma_cond_vol_test = np.sqrt(ewma_forecast.variance.shift(2).dropna())
    ewma_cond_vol_train.columns = ['cond_vol']
    ewma_cond_vol_test.columns = ['cond_vol']
    ewma_cond_vol = pd.concat([ewma_cond_vol_train, ewma_cond_vol_test], axis=0)
    ewma_resid = ret_data.shift(1).dropna() - ewma_res.params[0]

    ewma_var1 = ewma_res.params[1] * np.square(ewma_cond_vol)
    ewma_var1.columns = ['ewma_cond_vol']
    ewma_var2 = (1 - ewma_res.params[1]) * np.square(ewma_resid).to_frame()
    ewma_var2.columns = ['ewma_resid']

    var = pd.concat([garch_var1, garch_var2, egarch_var1, egarch_var2, egarch_var3, ewma_var1, ewma_var2], axis=1)
    return var


# %%
start_date = dt.datetime(2000, 1, 3)
split_date = dt.datetime(2013, 12, 31)

yy1 = spx_ret_data.loc[spx_ret_data.index > start_date]
yy2 = kospi_ret_data.loc[kospi_ret_data.index > start_date]

us_variables = generate_garch_variables(yy1, split_date=split_date)
kr_variables = generate_garch_variables(yy2, split_date=split_date)


# %%
us_variables.head()


# %%
kr_variables.head()

# %% [markdown]
# ## Rolling 

# %%
def run_garch_rolling(y, rvol, model, split_date, x=None, verbose=True, lam=None):

    # specify mean model
    ls = ConstantMean(y=y)
    
    # specify volatility model
    if model == "GARCH":
        ls.volatility = GARCH(p=1, q=1)
    elif model == "EGARCH":
        ls.volatility = EGARCH(p=1, o=1, q=1)
    elif model == "EWMA":
        ls.volatility = EWMAVariance(lam)
    else:
        print("Misspecified volatility process name")
    
    res = ls.fit(disp='off', last_obs=split_date)
    
    forecasts_1d = res.forecast(horizon=1)
    forecasted_vol = forecasts_1d.variance.pow(0.5).shift(1).dropna()
    
    test_merged = rvol.join(forecasted_vol).dropna()
    train_merged = rvol.join(res.conditional_volatility).dropna()

    test_MAE = np.abs(test_merged.iloc[:,0] - test_merged.iloc[:,1]).sum()
    train_MAE = np.abs(train_merged.iloc[:,0] - train_merged.iloc[:,1]).sum()
    MAE = [train_MAE, test_MAE]
    
    test_MSE = np.square(test_merged.iloc[:,0] - test_merged.iloc[:,1]).sum()
    train_MSE = np.square(train_merged.iloc[:,0] - train_merged.iloc[:,1]).sum()
    MSE = [train_MSE, test_MSE]
    
    test_HMAE = np.abs(1 - test_merged.iloc[:,1] / test_merged.iloc[:,0]).sum()
    train_HMAE = np.abs(1 - train_merged.iloc[:,1] / train_merged.iloc[:,0]).sum()
    HMAE = [train_HMAE, test_HMAE]
    
    test_HMSE = np.square(1 - test_merged.iloc[:,1] / test_merged.iloc[:,0]).sum()
    train_HMSE = np.square(1 - train_merged.iloc[:,1] / train_merged.iloc[:,0]).sum()
    HMSE = [train_HMSE, test_HMSE]

    df_results = pd.DataFrame(data=np.c_[MAE, MSE, HMAE, HMSE].T, columns=[model + ' ' + x for x in ['in-sample', 'out-of-sample']],                               index=['MAE', 'MSE', 'HMAE', 'HMSE']).T
    
    return df_results, len(train_merged), len(test_merged)


# %%
def get_rolling_results(df_return, rvol, start_date, split_date, window):

    start_pos = df_return.index.get_loc(start_date)
    end_pos = df_return.index.get_loc(split_date) + window + 1
    n = (df_return.index.get_loc(df_return.index[-1]) - df_return.index.get_loc(split_date)) / window
    garch_results = pd.DataFrame()
    egarch_results = pd.DataFrame()
    ewma_results = pd.DataFrame()
    sample_length = 0 
    
    for i in range(int(n)):
        
        y_temp = df_return.iloc[start_pos + i * window : end_pos + i * window]
        rvol_temp = rvol.iloc[start_pos + i * window : end_pos + i * window]
        split_date_temp = df_return.index[df_return.index.get_loc(split_date) + i * window]
        
        garch_results_temp_full = run_garch_rolling(y=y_temp*100, rvol=rvol_temp*100, model='GARCH', split_date=split_date_temp, verbose=False)
        garch_results_temp = garch_results_temp_full[0]
        sample_length += garch_results_temp_full[2]
        egarch_results_temp = run_garch_rolling(y=y_temp*100, rvol=rvol_temp*100, model='EGARCH', split_date=split_date_temp, verbose=False)[0]
        ewma_results_temp = run_garch_rolling(y=y_temp*100, rvol=rvol_temp*100, model='EWMA', split_date=split_date_temp, verbose=False)[0]
        
        garch_results = pd.concat([garch_results, garch_results_temp.iloc[1, :]], axis=1)
        egarch_results = pd.concat([egarch_results, egarch_results_temp.iloc[1, :]], axis=1)
        ewma_results = pd.concat([ewma_results, ewma_results_temp.iloc[1, :]], axis=1)
        
    results = pd.concat([garch_results.T.reset_index(drop=True), egarch_results.T.reset_index(drop=True), ewma_results.T.reset_index(drop=True)], axis=1)
    results.columns = pd.MultiIndex.from_product([['GARCH', 'EGARCH', 'EWMA'], results.columns[:4]])
    
    return results, sample_length

# %% [markdown]
# ### Out-of-sample results for rolling estimation
# #### (date index represents last day of test data)

# %%
us_rolling_result_raw, us_l = get_rolling_results(spx_ret_data, rvol=rvol1, start_date=dt.datetime(2000, 1, 4), split_date=dt.datetime(2013, 12, 31), window=22)
us_rolling_result = us_rolling_result_raw.sum(axis=0) / us_l


# %%
us_result_oos = us_result[us_result.index.str.endswith('out-of-sample')].stack()


# %%
us_result_oos.index = us_rolling_result.index


# %%
us_compare = pd.concat([us_rolling_result, us_result_oos], axis=1)
us_compare.columns = ['rolling', 'static']
us_compare


# %%
kr_rolling_result_raw, kr_l = get_rolling_results(kospi_ret_data, rvol=rvol2, start_date=dt.datetime(2000, 1, 4), split_date=dt.datetime(2013, 12, 30), window=22)
kr_rolling_result = kr_rolling_result_raw.sum(axis=0) / kr_l


# %%
kr_result_oos = kr_result[kr_result.index.str.endswith('out-of-sample')].stack()


# %%
kr_result_oos.index = kr_rolling_result.index


# %%
kr_compare = pd.concat([kr_rolling_result, kr_result_oos], axis=1)
kr_compare.columns = ['rolling', 'static']
kr_compare


# %%



