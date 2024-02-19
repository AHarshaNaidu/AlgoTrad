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

# Function to get ticker data
def get_ticker_data(tickerSymbol, start_date, end_date):
    tickerData = yf.Ticker(tickerSymbol)
    return tickerData.history(period='1d', start=start_date, end=end_date)

# Function to create LSTM model
def create_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Main function for Stock Analytics
def stock_analytics():
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

    # Ticker symbol selection
    tickerSymbol = st.sidebar.text_input('Enter Stock Ticker Symbol', 'AAPL')
    tickerDf = get_ticker_data(tickerSymbol, start_date, end_date)

    # Ticker information
    string_name = tickerDf.info.get('longName', 'N/A')
    st.header('**%s**' % string_name)

    string_summary = tickerDf.info.get('longBusinessSummary', 'N/A')
    st.info(string_summary)

    # Ticker data
    st.header('**Ticker data**')
    st.write(tickerDf)

    # Check if 'Close' column exists and there are enough data points
    if 'Close' in tickerDf.columns and len(tickerDf) > 1:
        # Daily Returns
        st.header('**Daily Returns**')
        tickerDf_cleaned = tickerDf.dropna()  # Drop rows with missing values
        daily_returns = tickerDf_cleaned['Close'].pct_change()
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
        tickerDf = calculate_ichimoku_cloud(tickerDf)

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
        seq_length = 60
        X, y = create_sequences(scaled_data, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Build the LSTM model
        model = create_lstm_model(X_train)

        # Compile and fit the model
        model.fit(X_train, y_train, epochs=20, batch_size=64)

        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Plot actual vs predicted prices
        st.header('**Actual vs Predicted Prices**')
        prediction_df = pd.DataFrame({'Actual': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), 'Predicted': predictions.flatten()})
        st.write(prediction_df)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=np.arange(len(y_test)), y=scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), mode='lines', name='Actual'))
        fig_pred.add_trace(go.Scatter(x=np.arange(len(predictions)), y=predictions.flatten(), mode='lines', name='Predicted'))
        fig_pred.update_layout(title='Actual vs Predicted Prices')
        st.plotly_chart(fig_pred)
    else:
        st.error("Failed to compute daily returns. Please check if the 'Close' column exists and there are enough data points.")

# Portfolio Optimization page
def portfolio_optimization():
    st.title("Portfolio Optimization")

    # Sidebar for portfolio optimization
    st.sidebar.subheader('Portfolio Optimization')
    start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
    end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

    tickerSymbols = st.sidebar.text_input('Enter Stock Ticker Symbols (comma-separated)', 'AAPL, MSFT, GOOGL')
    tickers = [x.strip() for x in tickerSymbols.split(',')]
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    if not data.empty:
        st.header('**Selected Ticker Data**')
        st.write(data)

        cleaned_weights = optimize_portfolio(data)

        st.header('**Optimized Portfolio Weights**')
        st.write(pd.Series(cleaned_weights))
    else:
        st.error("No data available for selected tickers. Please check your input.")

# App navigation
if __name__ == '__main__':
    pages = {
        "Stock Analytics": stock_analytics,
        "Portfolio Optimization": portfolio_optimization
    }

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()
