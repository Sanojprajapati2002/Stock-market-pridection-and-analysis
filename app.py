import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader as data
from tensorflow import keras
from keras.models import load_model
import streamlit as st

st.set_page_config(page_title="Stock Prediction and Analysis",page_icon=":bar_chart:", layout="wide")
st.title('Stock Prediction and Analysis')
user_input =st.text_input('Enter Stock Ticker', 'ZEEL.NS ')
df = yf.download(tickers=user_input,period='1y', interval='1D')

#describing data
st.subheader('5 years Data')
st.write(df.describe())
st.write('_____')

#visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close)
st.pyplot(fig)
st.write('_____')

st.subheader('Closing Price vs Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)
st.write('_____')

st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'b')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'r')
st.pyplot(fig)

#split data into training and testing
data_training =pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing =pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape) 
print(data_testing.shape)

#minmax Scaling
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
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test =y_test * scale_factor
st.write('_____')

#final
st.subheader('Predicted vs Original Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label= 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#removing watermark
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)