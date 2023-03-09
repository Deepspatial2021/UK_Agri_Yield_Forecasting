# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:26:47 2023

@author: asus
"""

import pandas as pd
import numpy as np
import prophet
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import acf, pacf_ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
import matplotlib.pyplot as plt
from scipy import stats as stats
def mape_scoring(model,X,y):
    pred=model.predict(X)
    mape=np.mean(np.abs((y - pred) /y)) * 100
    return mape

from sklearn.model_selection import cross_val_score



#%%

df=pd.read_excel("Crop Yield dataset.xlsx")

rice_yield=(pd.DataFrame(df.iloc[0,7:])).T
cols_drop=[col for col in rice_yield.columns if col.endswith("F")]

rice_yield=rice_yield.drop(columns=cols_drop)
rice_yield.columns=[int(col[1:]) for col in rice_yield.columns]
rice_yield=(rice_yield.T.reset_index()).rename(columns={"index":"Year",0:"Yield"})
rice_yield['Year']=pd.to_datetime(rice_yield['Year'],format="%Y")
rice_yield['Yield']=pd.to_numeric(rice_yield['Yield']/10000)
rice_yield_ts=rice_yield.set_index('Year')

rice_yield_ts.plot()


#%%

# Prophet:
    
rice_yield.columns = ['ds', 'y']
train = rice_yield.drop(rice_yield.index[-12:])
print(train.tail())

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=12,freq="Y")
future.tail()# use the model to make a forecast
forecast = model.predict(future)


# calculate MAE between expected and predicted values for december
y_true = rice_yield['y'][-12:].values
y_pred = forecast['yhat'][-12:].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)

# Making Test Dataframe
prophet_df=pd.DataFrame(zip(y_true,y_pred),columns=['Actual','Prophet Prediction'],index=rice_yield_ts.index[-12:])
prophet_df.plot()

# MAPE
np.mean(np.abs((y_true-y_pred) / y_true)) * 100


# Predicting Future:
    
model_2 = Prophet()
model_2.fit(rice_yield)

future = model_2.make_future_dataframe(periods=5,freq="Y")
forecast = model_2.predict(future)
    
#forecast.to_excel("Train/Prophet model.xlsx")

###############################*************###############################

#%%

# ETS

def auto_hwm(timeseries, val_split_date, alpha=[None], beta=[None], gamma=[None], 
              trend=None, seasonal=None, periods=None, verbose=False):

    best_params = []
    actual = timeseries[val_split_date:]

    print('Evaluating Exponential Smoothing model for', len(alpha) * len(beta) * len(gamma), 'fits\n')

    for a in alpha:
        for b in beta:
            for g in gamma:

                    if(verbose == True):
                        print('Checking for', {'alpha': a, 'beta': b, 'gamma': g})

                    model = ExponentialSmoothing(timeseries, trend=trend, seasonal=seasonal, seasonal_periods=periods)
                    model.fit(smoothing_level=a, smoothing_slope=b, smoothing_seasonal=g)
                    f_cast = model.predict(model.params, start=actual.index[0])
                    score = np.where(np.float64(mean_absolute_error(actual, f_cast)/actual).mean()>0,np.float64(mean_absolute_error(actual, f_cast)/actual).mean(),0)

                    best_params.append({'alpha': a, 'beta': b, 'gamma': g, 'mae': score})

    return min(best_params, key=lambda x: x['mae'])

alpha = np.arange(0.1,0.9,0.1)
beta = np.arange(0.1,0.9,0.1)
gamma =np.arange(0.1,0.9,0.1)

res=auto_hwm(rice_yield_ts, val_split_date ='2014-04-01' , alpha=alpha, beta=beta, gamma=gamma, 
              trend='mul', seasonal='mul', periods=12, verbose=True)

#%%

train_data=rice_yield_ts[:'2007-04-01']
test_data=rice_yield_ts['2007-04-01':]
ets_model = ExponentialSmoothing(train_data, trend='mul', seasonal='mul', \
                                 seasonal_periods=12).fit(smoothing_level=0.1, \
                                smoothing_slope=0.5, smoothing_seasonal=0.1)

ets_fc=ets_model.forecast(12)
ets_fc_df=pd.DataFrame(zip(test_data['Yield'],ets_fc),columns=['Actual','ETS Prediction'],index=test_data.index)

np.mean(np.abs((ets_fc_df['Actual'] - ets_fc_df['ETS Prediction']) / ets_fc_df['Actual'] )) * 100

ets_fc_df.plot()
train_ets=pd.DataFrame(zip(train_data['Yield'],ets_model.fittedvalues),columns=['Actual','Predicted'],index=train_data.index)


# Predicting Future:
ets_model_2 = ExponentialSmoothing(rice_yield_ts, trend='mul', seasonal='mul', \
                                 seasonal_periods=12).fit(smoothing_level=0.1, \
                                smoothing_slope=0.5, smoothing_seasonal=0.1)
                                                          
ets_fc_2=ets_model_2.forecast(5)
    

###################*******************##############################

#%%
# ARIMA model

adfuller(rice_yield_ts.diff().dropna())

inp_ts=rice_yield_ts.diff().dropna()

smt.graphics.plot_pacf(inp_ts,lags=10) # p=1
smt.graphics.plot_acf(inp_ts,lags=10) # q=1


from statsmodels.tsa.arima.model import ARIMA

arima_model=ARIMA(train_data,order=(0,1,1))
res=arima_model.fit()
print(res.summary())

arima_fc_values=res.forecast(12)
arima_fc_df=pd.DataFrame(zip(test_data['Yield'],arima_fc_values),columns=['Actual','Predicted'],index=test_data.index)

np.mean(np.abs((arima_fc_df['Actual'] - arima_fc_df['Predicted']) / arima_fc_df['Actual'] )) * 100
arima_fc_df.plot()

#%%
# SARIMA

import statsmodels as sm
sarima_model = sm.tsa.statespace.api.SARIMAX(train_data, order = (0,1,1), seasonal_order= (2,1,0,12),
                                             enforce_stationarity=False,
                                             enforce_invertibility=False).fit()

print(sarima_model.summary())

sarima_fc_val=sarima_model.forecast(12)
sarima_fc_df=pd.DataFrame(zip(test_data['Yield'],sarima_fc_val),columns=['Actual','Predicted'],index=test_data.index)

np.mean(np.abs((sarima_fc_df['Actual'] - sarima_fc_df['Predicted']) / sarima_fc_df['Actual'] )) * 100
sarima_fc_df.plot()

# Train Dataframe
train_sarima=pd.DataFrame(zip(train_data['Yield'],sarima_model.fittedvalues),columns=['Actual','Predicted'],index=train_data.index)


# Future Prediction
sarima_model_2 = sm.tsa.statespace.api.SARIMAX(rice_yield_ts, order = (0,1,1), seasonal_order= (2,1,0,12),
                                             enforce_stationarity=False,
                                             enforce_invertibility=False).fit()
sarima_fc_val_2=sarima_model_2.forecast(5)




#%%

# PMDARIMA

from pmdarima import auto_arima

auto_model=auto_arima(train_data,start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=12, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=20,n_fits=50)

auto_model.summary()

auto_prediction = auto_model.predict(n_periods = 12)
auto_df=pd.DataFrame(zip(test_data['Yield'],auto_prediction),columns=['Actual','Predicted'],index=test_data.index)

np.mean(np.abs((auto_df['Actual'] - auto_df['Predicted']) / auto_df['Actual'] )) * 100
auto_df.plot()

# Future Predict
auto_model_2=auto_arima(rice_yield_ts,start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=12, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=20,n_fits=50)

auto_prediction_2 = auto_model_2.predict(n_periods = 5)
auto_df_2=pd.DataFrame(zip(auto_prediction_2),columns=['Predicted'],index=sarima_fc_val_2.index)


#%%


# Linear Regression

# Taking lag for Prediction
lag_year=[]    
for n in list(range(1,9)):
    correl_df=pd.DataFrame(zip(rice_yield['y'],rice_yield['y'].shift(n)),columns=['Original','Shifted'])
    correl_df.dropna(inplace=True)
    lag_year.append([n,stats.pearsonr(correl_df['Original'],correl_df['Shifted']).statistic])

lag=2


# Adding Rolling mean feature
rolling_year=[]
for r in list(range(2,10)):
    rolling_df=pd.DataFrame(zip(rice_yield['y'],rice_yield['y'].shift(2).rolling(r,closed='left').mean()),columns=['Original','Rolling Average'])
    rolling_df.dropna(inplace=True)
    rolling_year.append([r,stats.pearsonr(rolling_df['Original'],rolling_df['Rolling Average']).statistic])
    
rolling_year=4   


#%%

rice_yield['Lag']=rice_yield['y'].shift(2)
rice_yield['Rolling Average']=rice_yield['y'].shift(2).rolling(4,closed='left').mean()

ts_data=rice_yield.set_index('ds')
inp_data=ts_data.dropna()

from sklearn.model_selection import train_test_split


X=inp_data[inp_data.columns.difference(['y'])]
y=inp_data['y']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

lr=LinearRegression()
lr.fit(X_train,y_train)
preds=lr.predict(X_test)
np.mean(np.abs((y_test - preds) / y_test)) * 100

pred_test=lr.predict(X['2007-04-01':])
act_test=y['2007-04-01':]
np.mean(np.abs((act_test - pred_test) / act_test)) * 100


lr_df=pd.DataFrame(zip(act_test,pred_test),columns=['Actual','Predicted'],index=test_data.index)
scores = cross_val_score(lr, X, y, cv=20,scoring=mape_scoring,n_jobs=-1)
print('Cross Validated MAPE median =',(pd.Series(scores)).median())
print('Cross Validated MAPE mean =',scores.mean())

# Train Predictions
train_lr_df=pd.DataFrame(zip(train_data['Yield'][6:],lr.predict(X[:"2007-04-01"])),columns=['Actual','Predicted'],
                         index=train_data.index[6:])

# Future Predictions
lag_series=rice_yield['y'].shift(2)
roll_avg_series=rice_yield['y'].rolling(4,closed='left').mean()

lr_future=pd.concat([rice_yield,pd.DataFrame(index=sarima_fc_val_2.index)])

lr_future['Lag']=lr_future['y'].shift(2)
lr_future['Rolling Average']=lr_future['y'].shift(2).rolling(4,closed='left').mean()

future_X=lr_future.iloc[[-5,-4],[2,3]]
future_lr_pred=lr.predict(future_X)
future_lr_df=pd.DataFrame(zip(future_lr_pred),columns=['Predicted'],index=future_X.index)


#%%

# Test Set Predictions

# with pd.ExcelWriter("Test Set Model Predictions.xlsx") as writer:
#     prophet_df.to_excel(writer,sheet_name="Prophet")
#     sarima_fc_df.to_excel(writer,sheet_name="SARIMA")
#     auto_df.to_excel(writer,sheet_name="AUTO ARIMA")
#     ets_fc_df.to_excel(writer,sheet_name="ETS")
#     lr_df.to_excel(writer,sheet_name="Linear Regression")


# Train Set Predictions

# with pd.ExcelWriter("Train/Train Set Model Predictions.xlsx") as writer:
#       train_ets.to_excel(writer,sheet_name="ETS")
#       train_sarima.to_excel(writer,sheet_name="SARIMA")
#       train_lr_df.to_excel(writer,sheet_name="Linear Regression")



# Future Predictions

# with pd.ExcelWriter("Future Predictions.xlsx") as writer_future:
#     ets_fc_2.to_excel(writer_future,sheet_name="ETS")
#     sarima_fc_val_2.to_excel(writer_future,sheet_name="SARIMA")
#     auto_df_2.to_excel(writer_future,sheet_name="AUTO ARIMA")
#     future_lr_df.to_excel(writer_future,sheet_name="Linear Regression")






