# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
# from keras.optimizers import SGD
# import math
# from sklearn.metrics import mean_squared_error
# import time
# # import datetime
# from datetime import datetime
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.preprocessing import LabelEncoder
# from keras.callbacks import EarlyStopping

# #train the model
# # model = tf.keras.models.load_model('/content/drive/MyDrive/CE Sem 6/SDP/Models/final_model_h5.h5')
# model = Sequential()

# FEATURES=['Close', 'Company']
# sequence_length=50
# dataset = pd.read_csv('./dataset_new.csv')

# data_filtered=dataset[FEATURES]
# print(data_filtered)
# nrows = data_filtered.shape[0]
# # print(nrows)
# data_filtered_ext = data_filtered.copy()
# data_filtered_ext['Prediction'] = data_filtered_ext['Close']
# # print(data_filtered_ext)

#   # Convert the data to numpy values
# np_data_unscaled = np.array(data_filtered)
# np_data = np.reshape(np_data_unscaled, (nrows, -1))
# scaler = MinMaxScaler()
# np_data_scaled = scaler.fit_transform(np_data_unscaled)

#   # print(np_data.shape)

# scaler_pred = MinMaxScaler()
# df_Close = pd.DataFrame(data_filtered_ext['Close'])
# np_Close_scaled = scaler_pred.fit_transform(df_Close)
#   # print(dataset)

# index_Close = data_filtered.columns.get_loc("Close")

#   # scaler = MinMaxScaler()
#   # np_data_scaled = scaler.fit_transform(np_data_unscaled)
#   # print(np_data_scaled)

#   # data_len = dataset.shape[0]
#   # print(dataset[data_len:data_len,:])


# train_data_len = math.ceil(np_data_scaled.shape[0] *0.8)
# train_data = np_data_scaled
# test_data = np_data_scaled[train_data_len - sequence_length:, :]

# def partition_dataset(sequence_length, data):
#       x, y = [], []
#       data_len = data_filtered.shape[0]
#       # for i in range(sequence_length, data_len):
#       x.append(data_filtered.iloc[data_len-sequence_length:data_len,:]) #contains sequence_length values 0-sequence_length * columsn
#       y.append(data_filtered.iloc[0, index_Close]) #contains the prediction values for validation,  for single-step prediction
      
#       # Convert the x and y to numpy arrays
#       x = np.array(x)
#       y = np.array(y)
#       return x, y


# X_train, y_train = partition_dataset(sequence_length, train_data)
# X_test, y_test = partition_dataset(sequence_length, test_data)


# # Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
# n_neurons = X_train.shape[1] * X_train.shape[2]
# print(n_neurons, X_train.shape[1], X_train.shape[2])
# model.add(LSTM(n_neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) 
# model.add(LSTM(n_neurons, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(5))
# model.add(Dropout(0.2))
# model.add(Dense(5))
# model.add(Dropout(0.2))
# model.add(Dense(1))

# # # Compile the model
# model.compile(optimizer='adam', loss='mse')

# # Training the model
# epochs = 50
# batch_size = 16
# early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
# history = model.fit(X_train, y_train, 
#                     batch_size=batch_size, 
#                     epochs=epochs,
#                     validation_data=(X_test, y_test)
#                    )