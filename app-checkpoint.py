import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader as data
#import keras.models 
from tensorflow import keras
model = keras.models.load_model('path/to/location')
#import load_model
import streamlit as st

stock_symbol = "ZEEL.NS"

df = yf.download(tickers=stock_symbol,C='1y', interval='1D')

#describing data
st.subheader('5 years')
st.write(df.describe())

#visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.fiure(figsize =(12,6))
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'B')
plt.plot(ma200, 'G')
plt.plot(df.close, 'R')
st.pyplot(fig)

#split data into training and testing
data_training =pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing =pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Load Model
model = load_model('keras_model.h5')

#Testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# making prediction
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/[0]
y_predicted = y_predicted * scale_factor
y_test =y_test * scale_factor

#final
st.subheader('Predicted vs Original Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label= 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)