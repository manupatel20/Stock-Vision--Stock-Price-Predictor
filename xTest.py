import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import time
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping

# dataset=pd.read_csv('/content/drive/MyDrive/CE Sem 6/SDP/Project_Sample_2/dataset.csv',  index_col='Date', parse_dates=['Date'])






def test(index):
  c_index=index

  dataset = pd.read_csv('./dataset.csv', index_col='Date', parse_dates=['Date'])

  dataset.pop(dataset.columns[0])

  FEATURES = [ 'Close','Company']

  dataset=dataset[FEATURES]

  le = LabelEncoder()
  dataset['Company'] = le.fit_transform(dataset['Company'])

  print(dataset.tail())
  print(dataset.info())

  model= tf.keras.models.load_model('./final_model_h5.h5')

  data_filtered = dataset[FEATURES]
  data_filtered_ext = data_filtered.copy()
  data_filtered_ext['Prediction'] = data_filtered_ext['Close']
  # print(data_filtered_ext)

  nrows = data_filtered.shape[0]

  # Convert the data to numpy values
  np_data_unscaled = np.array(data_filtered)
  np_data = np.reshape(np_data_unscaled, (nrows, -1))
  print(np_data.shape)

  # Transform the data by scaling each feature to a range between 0 and 1
  scaler = MinMaxScaler()
  np_data_scaled = scaler.fit_transform(np_data_unscaled)

  # Creating a separate scaler that works on a single column for scaling predictions
  scaler_pred = MinMaxScaler()
  df_Close = pd.DataFrame(data_filtered_ext['Close'])
  np_Close_scaled = scaler_pred.fit_transform(df_Close)

  sequence_length = 50

  # Prediction Index
  index_Close = dataset.columns.get_loc("Close")

  print(np_data)

  from datetime import datetime

  currentDay = 17
  prevDay = currentDay-1
  currentMonth = 3
  currentYear = 2023

  if(currentDay == 1):
    if(currentMonth==2):
      prevDay=28
      currentMonth-=1
    elif(currentMonth % 2 ==0):
      prevDay=30
      currentMonth-=1
    else:
      prevDay=31
      currentMonth-=1

  # df_temp =dataset.loc()
  df_temp =dataset.loc[dataset['Company'] == c_index]
  # print(df_temp)

  df_temp = df_temp[-sequence_length:]
  new_df = df_temp.filter(FEATURES)

  N = sequence_length

  # Get the last N day closing price values and scale the data to be values between 0 and 1
  last_N_days = new_df[-sequence_length:].values
  last_N_days_scaled = scaler.transform(last_N_days)

  # Create an empty list and Append past N days
  X_test_new = []
  X_test_new.append(last_N_days_scaled)
  return X_test_new