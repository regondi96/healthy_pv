#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.metrics import mean_squared_error,r2_score
## Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers import Dropout


# In[2]:


#Adapted from https://www.kaggle.com/gurpreetmohaar Thanks.
missing_values = ["0"]
data = pd.read_csv("/Users/hogar/Desktop/datasetNREL.csv", na_values = missing_values)


# In[3]:


data['Datetime'] = pd.to_datetime(data['Unnamed: 0'])
data = data.set_index('Datetime')
data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.drop(['Cloud Type'], axis=1, inplace=True)
data.drop(['Solar Zenith Angle'], axis=1, inplace=True)
data.drop(['Wind Speed'], axis=1, inplace=True)
data.drop(['Wind Direction'], axis=1, inplace=True)

data.head()


# In[4]:


clean_data = data.dropna()
clean_data.head()


# In[5]:


def get_acf(data,lags): 
    frame = []
    for i in range(lags+1):
        frame.append(data.apply(lambda col: col.autocorr(i), axis=0))
    return pd.DataFrame(frame).plot.line()
get_acf(data,10)


# In[6]:


#Its imperative from autocorealtion plots that we dont need each minute data, perhaps we can decide roll up the aggregate
i = 1
# plot each column
plt.figure(figsize=(20, 15))
for counter in range(1,len(clean_data.columns)):
    plt.subplot(len(clean_data.columns), 1, i)
    plt.plot(clean_data.resample('D').mean().values[:, counter], color = 'blue')
    plt.title(clean_data.columns[counter], y=0.8, loc='right')
    i = i+1
plt.show()


# In[7]:


#check corelation matrices for minute, hour and day
#minute
corr_min = clean_data.corr()
#hour
corr_hour = pd.DataFrame(clean_data.resample('H').mean().values).corr()
#day
corr_day = pd.DataFrame(clean_data.resample('D').mean().values).corr()
cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]
#Plot Minute
corr_min.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# In[8]:


resampled_data = clean_data.resample('H').mean() 
resampled_data.shape


# In[9]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[10]:


reframed_data = series_to_supervised(resampled_data, 1, 1)
print(reframed_data.head())


# In[11]:


reframed_data.drop(reframed_data.columns[[7,8,9,10,11]], axis=1, inplace=True)
print(reframed_data.columns)


# In[12]:


# split into train and test sets
values = reframed_data.values
train_index = 500*48 #The logic is to have 500 days worth of training data. this could also be a hyperparameter that can be tuned.
train = values[:train_index, :]
test = values[train_index:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].


# In[13]:


model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=10, batch_size=20, validation_data=(test_X, test_y), verbose=1, shuffle=False)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[14]:


yhat = model.predict(test_X, verbose=0)
rmse = np.sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)


# In[37]:


## time steps, every step is one hour (you can easily convert the time step to the actual time index)
## for a demonstration purpose, I only compare the predictions in 200 hours. 

aa=[x for x in range(100)]
plt.plot(aa, test_y[:100], marker='.', label = "Real")
plt.plot(aa, yhat[:100], 'r', label = "Prediction")
plt.ylabel('Temperature', size=15)
plt.xlabel('Time', size=15)
plt.legend(fontsize=15)
plt.show()


# In[ ]:




