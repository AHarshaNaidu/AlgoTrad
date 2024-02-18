import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
from ta.trend import IchimokuIndicator
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import cvxpy as cp  # Import cvxpy for portfolio optimization

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# App title
st.markdown('''
# Algorithmic Trading Strategies
Scale +91 Hackathon | FFI 2024
# Team GARUDA
Akula Sri Harsha Sri Sai Hanuman (LinkedIN.com/in/AHarshaNaidu)
''')
st.write('---')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker

# Ticker information
string_logo = ''
if 'logo_url' in tickerData.info:
    string_logo = '<img src=%s>' % tickerData.info['logo_url']
    st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info.get('longName', 'N/A')
st.header('**%s**' % string_name)

string_summary = tickerData.info.get('longBusinessSummary', 'N/A')
st.info(string_summary)

# Ticker data
st.header('**Ticker data**')
st.write(tickerDf)

# Daily Returns
st.header('**Daily Returns**')
daily_returns = tickerDf['Close'].pct_change()
st.write(daily_returns)

# Cumulative Returns
st.header('**Cumulative Returns**')
cumulative_returns = daily_returns.cumsum()
st.write(cumulative_returns)

# Bollinger bands
st.header('**Bollinger Bands**')
qf = cf.QuantFig(tickerDf, title='First Quant Figure', legend='top', name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

# Ichimoku Cloud
st.header('**Ichimoku Cloud**')

# Calculate Ichimoku Cloud data
indicator_ichimoku = IchimokuIndicator(high=tickerDf['High'], low=tickerDf['Low'])
tickerDf['ichimoku_a'] = indicator_ichimoku.ichimoku_a()
tickerDf['ichimoku_b'] = indicator_ichimoku.ichimoku_b()
tickerDf['ichimoku_base_line'] = indicator_ichimoku.ichimoku_base_line()
tickerDf['ichimoku_conversion_line'] = indicator_ichimoku.ichimoku_conversion_line()

# Plot Ichimoku Cloud
fig_ichimoku = go.Figure(data=[go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_a'], name='Ichimoku A'),
                               go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_b'], name='Ichimoku B'),
                               go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_base_line'], name='Base Line'),
                               go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_conversion_line'], name='Conversion Line')],
                         layout=go.Layout(title='Ichimoku Cloud'))
st.plotly_chart(fig_ichimoku)

# Stock Price Prediction using LSTM
st.header('**Stock Price Prediction using LSTM**')

# Prepare the data for prediction
data = tickerDf['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split data into training and testing sets
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# Build the LSTM model
model = Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=64),
    Dense(units=1)
])

# Compile and fit the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
batch_size = 128
num_batches = len(X_test) // batch_size
remainder = len(X_test) % batch_size

predictions = []
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch_predictions = model.predict(X_test[start_idx:end_idx])
    predictions.extend(batch_predictions)

# Predict the remaining data points
if remainder:
    batch_predictions = model.predict(X_test[-remainder:])
    predictions.extend(batch_predictions)

# Convert predictions to numpy array
predictions = np.array(predictions)

# Inverse transform the predictions
predictions = scaler.inverse_transform(predictions)

# Filter actual data to only include dates up to the last date of the training data
last_train_date = tickerDf.index[-len(X_test) - 1]
actual_data_filtered = tickerDf.loc[tickerDf.index <= last_train_date]

# Plot actual vs predicted prices
st.header('**Actual vs Predicted Prices**')

# Plot actual vs predicted prices
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=actual_data_filtered.index, y=actual_data_filtered['Close'], mode='lines', name='Actual'))
fig_pred.add_trace(go.Scatter(x=actual_data_filtered.index[-len(predictions):], y=predictions.flatten(), mode='lines', name='Predicted'))
fig_pred.update_layout(title='Actual vs Predicted Prices')
st.plotly_chart(fig_pred)
