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
df_temp =dataset.loc[dataset['Company'] == 3]
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

# Convert the X_test data set to a numpy array and reshape the data
pred_price_scaled = model.predict(np.array(X_test_new))
pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))
print(pred_price_unscaled)

stockname = 'TSLA'
# print(new_df)

ticker1='TSLA'
period1 = int(time.mktime(datetime(currentYear, currentMonth, prevDay, 23, 59).timetuple())) # year,month,date, hour, min
period2 = int(time.mktime(datetime(currentYear, currentMonth, currentDay, 23, 59).timetuple()))
interval = '1d'
query_string1 = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker1}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
df1 = pd.read_csv(query_string1)
df1.insert(loc=5,
          column='Company',
          value=3) # 3-TSLA
df1.to_csv('temp-dataset.csv')
df1=pd.read_csv('temp-dataset.csv',  index_col='Date', parse_dates=['Date'])
df1.pop(df1.columns[0])
df1 = df1[FEATURES]
print(df1)

# Print last price and predicted price for the next day
# price_today = np.round(new_df['Close'][-1], 2)
predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
# change_percent = np.round(100 - (price_today * 100)/predicted_price, 2)

# end_date =  "2023-03-16"

plus = '+'; minus = ''
# print(f'The close price for {stockname} at {end_date} was {price_today}')
# print(f'The predicted close price is {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')

print(f'The predicted close price is {predicted_price}')

currentDay = datetime.now().day
currentMonth = datetime.now().month
currentYear = datetime.now().year

sequence_length = 50
prevDay=currentDay
prevMonth=currentMonth
prevYear = currentYear
upper_limit = 2*sequence_length

for i in range(1,upper_limit):
  prevDay = prevDay-1
  if(prevDay<1):
    if(currentMonth==3):
      prevDay=28
      prevMonth=2
    elif(currentMonth % 2 ==0):
      prevDay=31
      prevMonth-=1
    else:
      prevDay=31
      prevMonth-=1

    if(prevMonth<1):
      prevMonth=12
      prevYear-=1
  # print("year ",prevYear, " Month: ", prevMonth, " Day: ", prevDay)


# print(prevMonth)

ticker1='TSLA'
period1 = int(time.mktime(datetime(prevYear, prevMonth, prevDay, 1, 30).timetuple())) # year,month,date, hour, min
period2 = int(time.mktime(datetime(currentYear, currentMonth, currentDay, 23, 59).timetuple()))
interval = '1d'
query_string1 = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker1}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
df1 = pd.read_csv(query_string1)

# print(df1)
df1.insert(loc=5,
          column='Company',
          value=3)
df1.to_csv('temp_dataset.csv')

dataset=pd.read_csv('temp_dataset.csv',  index_col='Date', parse_dates=['Date'])
dataset.pop(dataset.columns[0])
Filtered_Features = ['Close', 'Company']

data_filtered=dataset[Filtered_Features]
print(data_filtered)
nrows = data_filtered.shape[0]
# print(nrows)
data_filtered_ext = data_filtered.copy()
data_filtered_ext['Prediction'] = data_filtered_ext['Close']
# print(data_filtered_ext)

# Convert the data to numpy values
np_data_unscaled = np.array(data_filtered)
np_data = np.reshape(np_data_unscaled, (nrows, -1))

scaler = MinMaxScaler()
np_data_scaled = scaler.fit_transform(np_data_unscaled)

# print(np_data.shape)

scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(data_filtered_ext['Close'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)
# print(dataset)

index_Close = data_filtered.columns.get_loc("Close")

# scaler = MinMaxScaler()
# np_data_scaled = scaler.fit_transform(np_data_unscaled)
# print(np_data_scaled)

# data_len = dataset.shape[0]
# print(dataset[data_len:data_len,:])


train_data_len = math.ceil(np_data_scaled.shape[0])
train_data = np_data_scaled[0:train_data_len, :]
def partition_dataset(sequence_length, data):
    x, y = [], []
    data_len = data_filtered.shape[0]
    # for i in range(sequence_length, data_len):
    x.append(data_filtered.iloc[data_len-sequence_length:data_len,:]) #contains sequence_length values 0-sequence_length * columsn
    y.append(data_filtered.iloc[0, index_Close]) #contains the prediction values for validation,  for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

x_train, y_train = partition_dataset(sequence_length, train_data)
# train_data = np_data_scaled[upper_limit-sequence_length:upper_limit,:]
# y_train = dataset[upper_limit-1,index_Close]

print(x_train.shape, y_train.shape)

model.fit(x_train, y_train)

# !pip install schedule

import schedule
import time

def daily_run():
  currentDay = datetime.now().day
  currentMonth = datetime.now().month
  currentYear = datetime.now().year

  sequence_length = 50
  prevDay=currentDay
  prevMonth=currentMonth
  prevYear = currentYear
  upper_limit = 2*sequence_length

  for i in range(1,upper_limit):
    prevDay = prevDay-1
    if(prevDay<1):
      if(currentMonth==3):
        prevDay=28
        prevMonth=2
      elif(currentMonth % 2 ==0):
        prevDay=31
        prevMonth-=1
      else:
        prevDay=31
        prevMonth-=1

      if(prevMonth<1):
        prevMonth=12
        prevYear-=1
    # print("year ",prevYear, " Month: ", prevMonth, " Day: ", prevDay)


  # print(prevMonth)

  ticker1='TSLA'
  period1 = int(time.mktime(datetime(prevYear, prevMonth, prevDay, 1, 30).timetuple())) # year,month,date, hour, min
  period2 = int(time.mktime(datetime(currentYear, currentMonth, currentDay, 23, 59).timetuple()))
  interval = '1d'
  query_string1 = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker1}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
  df1 = pd.read_csv(query_string1)

  # print(df1)
  df1.insert(loc=5,
            column='Company',
            value=3)
  df1.to_csv('temp_dataset.csv')

  dataset=pd.read_csv('temp_dataset.csv',  index_col='Date', parse_dates=['Date'])
  dataset.pop(dataset.columns[0])
  Filtered_Features = ['Close', 'Company']

  data_filtered=dataset[Filtered_Features]
  print(data_filtered)
  nrows = data_filtered.shape[0]
  # print(nrows)
  data_filtered_ext = data_filtered.copy()
  data_filtered_ext['Prediction'] = data_filtered_ext['Close']
  # print(data_filtered_ext)

  # Convert the data to numpy values
  np_data_unscaled = np.array(data_filtered)
  np_data = np.reshape(np_data_unscaled, (nrows, -1))

  scaler = MinMaxScaler()
  np_data_scaled = scaler.fit_transform(np_data_unscaled)

  # print(np_data.shape)

  scaler_pred = MinMaxScaler()
  df_Close = pd.DataFrame(data_filtered_ext['Close'])
  np_Close_scaled = scaler_pred.fit_transform(df_Close)
  # print(dataset)

  index_Close = data_filtered.columns.get_loc("Close")

  # scaler = MinMaxScaler()
  # np_data_scaled = scaler.fit_transform(np_data_unscaled)
  # print(np_data_scaled)

  # data_len = dataset.shape[0]
  # print(dataset[data_len:data_len,:])


  train_data_len = math.ceil(np_data_scaled.shape[0])
  train_data = np_data_scaled[0:train_data_len, :]
  def partition_dataset(sequence_length, data):
      x, y = [], []
      data_len = data_filtered.shape[0]
      # for i in range(sequence_length, data_len):
      x.append(data_filtered.iloc[data_len-sequence_length:data_len,:]) #contains sequence_length values 0-sequence_length * columsn
      y.append(data_filtered.iloc[0, index_Close]) #contains the prediction values for validation,  for single-step prediction
      
      # Convert the x and y to numpy arrays
      x = np.array(x)
      y = np.array(y)
      return x, y

  x_train, y_train = partition_dataset(sequence_length, train_data)
  # train_data = np_data_scaled[upper_limit-sequence_length:upper_limit,:]
  # y_train = dataset[upper_limit-1,index_Close]

  print(x_train.shape, y_train.shape)

  model.fit(x_train, y_train)

schedule.every().day.at("23:21").do(daily_run)
# schedule.every(10).seconds.do(daily_run)

while 1:
  schedule.run_pending()
  time.sleep(1)

  ###Prediction for 30 days but not accurate
# print(dataset)
import datetime
currentDay = 17
currentMonth = 3
currentYear = 2023

for i in range(30):
  x = datetime.datetime(currentYear, currentMonth, currentDay,0,0,0)
  print(x)
  N = sequence_length

  # Get the last N day closing price values and scale the data to be values between 0 and 1
  last_N_days = new_df[-sequence_length:].values
  last_N_days_scaled = scaler.transform(last_N_days)

  # Create an empty list and Append past N days
  X_test_new = []
  X_test_new.append(last_N_days_scaled)

  # Convert the X_test data set to a numpy array and reshape the data
  pred_price_scaled = model.predict(np.array(X_test_new))
  pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))
  predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
  test_set = pd.DataFrame({"Close": [predicted_price], "Company": [3], "Date":[x]})
  test_set.set_index('Date', inplace=True)
  print(test_set)

  frames=[new_df , test_set]
  new_df = pd.concat(frames)
  
  #update date
  currentDay+=1
  if((currentMonth == 2) and (currentDay>28)):
    currentMonth=3
    currentDay=1
  elif((currentMonth % 2 == 0) and (currentDay>30)):
    currentMonth+=1
    currentDay=1
  elif((currentMonth % 2 != 0) and (currentDay>31)):
    currentMonth+=1
    currentDay=1
  if(currentMonth == 12 and currentDay>31):
    currentDay=1
    currentMonth=1
    currentYear+=1


### successfully added 1 record in 'result_df' dataset, now iterate above procedure for 30 days and the make a new dataframe for it and then display it on graph
new_df["Close"][-sequence_length:].plot(figsize=(16,4),legend=True)
# dataset["Close"]['2022':].plot(figsize=(16,4),legend=True)
plt.legend(['Predicted price'])
plt.title('stock price')
plt.show()

#Actual code

from datetime import datetime

currentDay = datetime.now().day
prevDay = currentDay-1
currentMonth = datetime.now().month
currentYear = datetime.now().year

if(currentDay == 1):
  if(currentMonth==2):
    prevDay=28
  elif(currentMonth % 2 ==0):
    prevDay=30
  else:
    prevDay=31

df_temp =dataset.loc[dataset['Company'] == 3]
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

# Convert the X_test data set to a numpy array and reshape the data
pred_price_scaled = model.predict(np.array(X_test_new))
pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))
print(pred_price_unscaled)

stockname = 'TSLA'
# print(new_df)

ticker1='TSLA'
period1 = int(time.mktime(datetime(currentYear, currentMonth, prevDay, 23, 59).timetuple())) # year,month,date, hour, min
period2 = int(time.mktime(datetime(currentYear, currentMonth, currentDay, 23, 59).timetuple()))
interval = '1d'
query_string1 = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker1}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
df1 = pd.read_csv(query_string1)
df1.insert(loc=5,
          column='Company',
          value=3) # 3-TSLA
df1.to_csv('temp-dataset.csv')
df1=pd.read_csv('temp-dataset.csv',  index_col='Date', parse_dates=['Date'])
df1.pop(df1.columns[0])
df1 = df1[FEATURES]
print(df1)

# Print last price and predicted price for the next day
# price_today = np.round(new_df['Close'][-1], 2)
predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
# change_percent = np.round(100 - (price_today * 100)/predicted_price, 2)

# end_date =  "2023-03-16"

plus = '+'; minus = ''
# print(f'The close price for {stockname} at {end_date} was {price_today}')
# print(f'The predicted close price is {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')

print(f'The predicted close price is {predicted_price}')

dataset["Close"][:'2023-02'].plot(figsize=(16,4),legend=True)
test_set["Close"].plot(figsize=(16,4),legend=True)
plt.legend(['Training set (Before 2023)','Test set (2023 and beyond)'])
plt.title('stock price')
plt.show()