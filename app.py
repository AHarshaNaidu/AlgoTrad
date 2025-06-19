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
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(
    page_title="trAIde",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

about_content = """
# trAIde

developed by: [HARSHA](https://www.linkedin.com/in/AHarshaNaidu)

Welcome to trAIde, a cutting-edge trading decision making platform!

1. **Analyze Stocks**: Use indicators like Bollinger Bands and Ichimoku Cloud.
2. **Predict Stock Prices**: LSTM-based predictions.
"""

selected_tab = st.sidebar.radio("Select", ("About", "Stock Analysis", "Stock Price Prediction"))

if selected_tab == "About":
    st.markdown(about_content)

elif selected_tab == "Stock Analysis":
    st.sidebar.header('Stock Analysis Parameters')
    tickerSymbol = st.sidebar.text_input('Enter Stock Ticker Symbol', 'AAPL')
    tickerData = yf.Ticker(tickerSymbol)
    string_name = tickerData.info.get('longName', 'N/A')
    string_summary = tickerData.info.get('longBusinessSummary', 'N/A')

    st.subheader(f"Stock Analysis: {tickerSymbol} - {string_name}")
    st.info(string_summary)

    start_date = st.sidebar.date_input("Start Date", datetime.date(2019, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date(2021, 1, 31))
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

    st.header('Historical Stock Data')
    st.write(tickerDf)

    if 'Close' in tickerDf.columns and len(tickerDf) > 1:
        st.header('Daily Returns')
        daily_returns = tickerDf['Close'].pct_change()
        st.write(daily_returns)

        st.header('Cumulative Returns')
        cumulative_returns = daily_returns.cumsum()
        st.write(cumulative_returns)

        st.header('Bollinger Bands')
        qf = cf.QuantFig(tickerDf, title='Bollinger Bands', legend='top', name='GS')
        qf.add_bollinger_bands()
        fig = qf.iplot(asFigure=True)
        st.plotly_chart(fig)

        st.header('Ichimoku Cloud')
        indicator_ichimoku = IchimokuIndicator(high=tickerDf['High'], low=tickerDf['Low'])
        tickerDf['ichimoku_a'] = indicator_ichimoku.ichimoku_a()
        tickerDf['ichimoku_b'] = indicator_ichimoku.ichimoku_b()
        tickerDf['ichimoku_base_line'] = indicator_ichimoku.ichimoku_base_line()
        tickerDf['ichimoku_conversion_line'] = indicator_ichimoku.ichimoku_conversion_line()

        fig_ichimoku = go.Figure(data=[
            go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_a'], name='Ichimoku A'),
            go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_b'], name='Ichimoku B'),
            go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_base_line'], name='Base Line'),
            go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_conversion_line'], name='Conversion Line')
        ])
        st.plotly_chart(fig_ichimoku)
    else:
        st.error("Insufficient data or missing 'Close' column.")

elif selected_tab == "Stock Price Prediction":
    st.sidebar.header('Stock Prediction Parameters')
    tickerSymbol = st.sidebar.text_input('Enter Stock Ticker Symbol', 'AAPL')
    tickerData = yf.Ticker(tickerSymbol)
    string_name = tickerData.info.get('longName', 'N/A')

    st.subheader(f"Stock Price Prediction: {tickerSymbol} - {string_name}")
    start_date = st.sidebar.date_input("Start Date", datetime.date(2019, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date(2021, 1, 31))
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

    st.header('Historical Stock Data')
    st.write(tickerDf)

    if 'Close' in tickerDf.columns and len(tickerDf) > 1:
        st.header('Stock Price Prediction using LSTM')

        data = tickerDf['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        seq_length = 60
        X, y = create_sequences(scaled_data, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=64)

        predictions = scaler.inverse_transform(model.predict(X_test))
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        st.subheader('Model Evaluation')
        st.write(f"MSE: {mean_squared_error(actual, predictions):.2f}")
        st.write(f"MAE: {mean_absolute_error(actual, predictions):.2f}")

        st.header('Actual vs Predicted Prices')
        fig = go.Figure([
            go.Scatter(y=actual.flatten(), name='Actual'),
            go.Scatter(y=predictions.flatten(), name='Predicted')
        ])
        st.plotly_chart(fig)
    else:
        st.error("Insufficient data or missing 'Close' column.")
