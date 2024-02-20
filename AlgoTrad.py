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

# Set custom theme colors
PRIMARY_COLOR = "#E63946"  # Red
BACKGROUND_COLOR = "#F1FAEE"  # Light green
TEXT_COLOR = "#264653"  # Dark blue

# Set page configuration
st.set_page_config(
    page_title="Algorithmic Trading Strategies",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title('Algorithmic Trading Strategies')
st.markdown('---')
st.markdown('Scale +91 Hackathon | FFI 2024')
st.markdown('Team GARUDA')
st.write("Developed by: Akula Sri Harsha Sri Sai Hanuman ([LinkedIn](https://www.linkedin.com/in/AHarshaNaidu))")
st.write("This app provides various algorithmic trading strategies including technical analysis, stock price prediction using LSTM, and portfolio optimization.")

st.sidebar.title("Select Analysis")
option = st.sidebar.radio("", ("Stock Analysis", "Stock Price Prediction", "Portfolio Optimization"))

# Function to visualize returns with arrows
def visualize_returns(returns):
    arrows = ['↑' if r >= 0 else '↓' for r in returns]
    colors = ['green' if r >= 0 else 'red' for r in returns]
    return arrows, colors

# Stock Analysis
if option == "Stock Analysis":
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

    # Check if 'Close' column exists and there are enough data points
    if 'Close' in tickerDf.columns and len(tickerDf) > 1:
        # Display Returns Analysis
        st.header('Returns Analysis')

        # Daily Returns
        st.subheader('Daily Returns')
        daily_returns = tickerDf['Close'].pct_change()
        arrows, colors = visualize_returns(daily_returns)
        tickerDf['Daily Returns'] = daily_returns.map('{:.2%}'.format)
        tickerDf['Directional Indicator'] = arrows
        st.dataframe(tickerDf[['Daily Returns', 'Directional Indicator']].style.apply(lambda x: ['color: green' if v == '↑' else 'color: red' for v in x]))

        # Cumulative Returns
        st.subheader('Cumulative Returns')
        cumulative_returns = daily_returns.cumsum()
        cumulative_returns_with_arrows, cumulative_returns_colors = visualize_returns(cumulative_returns)
        tickerDf['Cumulative Returns'] = cumulative_returns.map('{:.2%}'.format)
        tickerDf['Directional Indicator (Cumulative)'] = cumulative_returns_with_arrows
        st.dataframe(tickerDf[['Cumulative Returns', 'Directional Indicator (Cumulative)']].style.apply(lambda x: ['color: green' if v == '↑' else 'color: red' for v in x]))

        # Bollinger bands
        st.header('Bollinger Bands')
        qf = cf.QuantFig(tickerDf, title='Bollinger Bands', legend='top', name='GS')
        qf.add_bollinger_bands()
        fig = qf.iplot(asFigure=True)
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
elif option == "Stock Price Prediction":
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

# Portfolio Optimization
elif option == "Portfolio Optimization":
    st.sidebar.header('Portfolio Optimization Parameters')
    tickerSymbols = st.sidebar.text_input('Enter Stock Ticker Symbols (comma-separated)', 'AAPL, MSFT, GOOGL')

    # Fetching data for selected tickers
    tickers = [x.strip() for x in tickerSymbols.split(',')]
    data = yf.download(tickers)['Adj Close']

    # Check if data is available for selected tickers
    if not data.empty:
        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(data)
        Sigma = risk_models.sample_cov(data)

        # Perform portfolio optimization
        ef = EfficientFrontier(mu, Sigma)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        expected_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)

        # Display selected ticker data
        st.subheader('Selected Ticker Data')
        st.write(data)

        # Display optimized portfolio weights
        st.subheader('Optimized Portfolio Weights')
        st.write(pd.Series(cleaned_weights))

        # Plot Efficient Frontier
        st.subheader('Efficient Frontier')
        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(go.Scatter(x=np.sqrt(np.diag(Sigma)), y=mu, mode='markers', name=ticker))

        # Highlighting the optimized portfolio
        fig.add_trace(go.Scatter(x=[annual_volatility], y=[expected_return], mode='markers', marker=dict(size=15, color='red'), name='Optimized Portfolio'))

        fig.update_layout(title='Efficient Frontier',
                          xaxis_title='Annual Volatility',
                          yaxis_title='Expected Annual Return')
        st.plotly_chart(fig)

        # Display portfolio metrics
        st.subheader('Portfolio Metrics')
        st.write(f'Expected Annual Return: {expected_return:.2%}')
        st.write(f'Annual Volatility: {annual_volatility:.2%}')
        st.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')

    else:
        st.error("No data available for selected tickers. Please check your input.")
