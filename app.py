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
import pypfopt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set custom theme colors
base="dark"
primaryColor="#674019"
backgroundColor="#000000"
secondaryBackgroundColor="#996515"

# Set page configuration
st.set_page_config(
    page_title="trAIde",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# About page content
about_content = """
# trAIde

developed by: [HARSHA](https://www.linkedin.com/in/AHarshaNaidu) - Top 50 (Prototyping) in Scale +91 Hackathon at Fintech Festival India (FFI) 2024

"Welcome to trAIde, a cutting-edge trading decision making platform! With this tool, you can take your investment strategy to the next level.

Here's what you can do:

1. **Analyze Stocks**: Dive deep into historical stock data, spot market trends, and patterns using technical analysis indicators like Bollinger Bands and Ichimoku Cloud.

2. **Predict Stock Prices**: Want to know where a stock is headed? Input a stock ticker symbol, and our powerful machine learning model will predict its future prices based on historical data.

Our platform is user-friendly and packed with features to help you make informed investment decisions. Start exploring now and take control of your financial future!".
"""

# Sidebar menu
selected_tab = st.sidebar.radio("Select", ("About", "Stock Analysis", "Stock Price Prediction"))

# About page
if selected_tab == "About":
    st.markdown(about_content)
# Stock Analysis
elif selected_tab == "Stock Analysis":
    st.sidebar.header('Stock Analysis Parameters')
    tickerSymbol = st.sidebar.text_input('Enter Stock Ticker Symbol', 'AAPL')

    # Fetching ticker information
    tickerData = yf.Ticker(tickerSymbol)
    string_name = tickerData.info.get('longName', 'N/A')
    string_summary = tickerData.info.get('longBusinessSummary', 'N/A')

    st.subheader(f"Stock Analysis: {tickerSymbol} - {string_name}")
    st.info(string_summary)

    # Ticker data
    st.header('Historical Stock Data')
    start_date = st.sidebar.date_input("Start Date", datetime.date(2019, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date(2021, 1, 31))
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
    st.write(tickerDf)

    # Check if 'Close' column exists and there are enough data points
    if 'Close' in tickerDf.columns and len(tickerDf) > 1:
        # Display Daily Returns
        st.header('Daily Returns')
        daily_returns = tickerDf['Close'].pct_change()
        st.write(daily_returns)

        # Display Cumulative Returns
        st.header('Cumulative Returns')
        cumulative_returns = daily_returns.cumsum()
        st.write(cumulative_returns)

        # Bollinger bands
        st.header('Bollinger Bands')
        # Calculate Bollinger Bands manually
        window = 20
        std_dev = 2
        rolling_mean = tickerDf['Close'].rolling(window=window).mean()
        rolling_std = tickerDf['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['Close'], name='Close'))
        fig.add_trace(go.Scatter(x=tickerDf.index, y=rolling_mean, name='Moving Avg'))
        fig.add_trace(go.Scatter(x=tickerDf.index, y=upper_band, name='Upper Band', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=tickerDf.index, y=lower_band, name='Lower Band', line=dict(dash='dot')))

        fig.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

        # Ichimoku Cloud
        st.header('Ichimoku Cloud')

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

    else:
        st.error("Failed to compute returns. Please check if the 'Close' column exists and there are enough data points.")

# Stock Price Prediction
elif selected_tab == "Stock Price Prediction":
    st.sidebar.header('Stock Prediction Parameters')
    tickerSymbol = st.sidebar.text_input('Enter Stock Ticker Symbol', 'AAPL')

    # Fetching ticker information
    tickerData = yf.Ticker(tickerSymbol)
    string_name = tickerData.info.get('longName', 'N/A')

    st.subheader(f"Stock Price Prediction: {tickerSymbol} - {string_name}")

    # Ticker data
    st.header('Historical Stock Data')
    start_date = st.sidebar.date_input("Start Date", datetime.date(2019, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date(2021, 1, 31))
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
    st.write(tickerDf)

    # Check if 'Close' column exists and there are enough data points
    if 'Close' in tickerDf.columns and len(tickerDf) > 1:
        # Stock Price Prediction using LSTM
        st.header('Stock Price Prediction using LSTM')

        # Prepare the data for prediction
        data = tickerDf['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        seq_length = 60
        X, y = create_sequences(scaled_data, seq_length)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        # Compile and fit the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=64)

        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        # Display evaluation results
        st.subheader('Model Evaluation')
        st.write(f'Mean Squared Error (MSE): {mse:.2f}')
        st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

        # Plot actual vs predicted prices
        st.header('Actual vs Predicted Prices')
        prediction_df = pd.DataFrame({'Actual': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), 'Predicted': predictions.flatten()})
        st.write(prediction_df)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=np.arange(len(y_test)), y=scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), mode='lines', name='Actual'))
        fig_pred.add_trace(go.Scatter(x=np.arange(len(predictions)), y=predictions.flatten(), mode='lines', name='Predicted'))
        fig_pred.update_layout(title='Actual vs Predicted Prices')
        st.plotly_chart(fig_pred)

    else:
        st.error("Failed to compute returns. Please check if the 'Close' column exists and there are enough data points.")
