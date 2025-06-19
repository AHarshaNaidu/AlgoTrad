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
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set custom theme colors
base="dark"
primaryColor="#674019"
backgroundColor="#000000"
secondaryBackgroundColor="#996515"

st.set_page_config(
    page_title="trAIde",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# About content
about_content = """
# trAIde

developed by: [HARSHA](https://www.linkedin.com/in/AHarshaNaidu)

Welcome to trAIde, a cutting-edge trading decision making platform!

1. **Analyze Stocks**: Use indicators like Bollinger Bands and Ichimoku Cloud.
2. **Predict Stock Prices**: LSTM-based predictions.
3. **Optimize Portfolios**: Long-term and short-term risk-return optimization.
"""

selected_tab = st.sidebar.radio("Select", ("About", "Stock Analysis", "Stock Price Prediction", "Long-Term Portfolio Optimization", "Short-Term Portfolio Optimization"))

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
            go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_conversion_line'], name='Conversion Line')],
            layout=go.Layout(title='Ichimoku Cloud'))
        st.plotly_chart(fig_ichimoku)
    else:
        st.error("Failed to compute returns. Please check if the 'Close' column exists and there are enough data points.")

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
        scaler = MinMaxScaler(feature_range=(0, 1))
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

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=64)

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        st.subheader('Model Evaluation')
        st.write(f'Mean Squared Error (MSE): {mse:.2f}')
        st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

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

elif selected_tab == "Long-Term Portfolio Optimization":
    st.sidebar.header('Long-Term Portfolio Optimization Parameters')
    tickerSymbols = st.sidebar.text_input('Enter Stock Ticker Symbols (comma-separated)', 'AAPL, MSFT, GOOGL')
    tickers = [x.strip() for x in tickerSymbols.split(',')]

    try:
        data = yf.download(tickers)
        if len(tickers) == 1:
            data = data['Adj Close'].to_frame(tickers[0])
        else:
            data = data['Adj Close']

        mu = expected_returns.mean_historical_return(data)
        Sigma = risk_models.sample_cov(data)

        ef = EfficientFrontier(mu, Sigma)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        expected_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)

        st.subheader('Selected Ticker Data')
        st.write(data)

        st.subheader('Optimized Portfolio Weights')
        st.write(pd.Series(cleaned_weights))

        st.subheader('Efficient Frontier')
        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(go.Scatter(x=np.sqrt(np.diag(Sigma)), y=mu, mode='markers', name=ticker))
        fig.add_trace(go.Scatter(x=[annual_volatility], y=[expected_return], mode='markers', marker=dict(size=15, color='red'), name='Optimized Portfolio'))
        fig.update_layout(title='Efficient Frontier', xaxis_title='Annual Volatility', yaxis_title='Expected Annual Return')
        st.plotly_chart(fig)

        st.subheader('Portfolio Metrics')
        st.write(f'Expected Annual Return: {expected_return:.2%}')
        st.write(f'Annual Volatility: {annual_volatility:.2%}')
        st.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')

    except Exception as e:
        st.error(f"An error occurred: {e}")

elif selected_tab == "Short-Term Portfolio Optimization":
    st.sidebar.header('Short-Term Portfolio Optimization Parameters')
    tickerSymbols = st.sidebar.text_input('Enter Stock Ticker Symbols (comma-separated)', 'AAPL, MSFT, GOOGL')
    tickers = [x.strip() for x in tickerSymbols.split(',')]

    try:
        data = yf.download(tickers)
        if len(tickers) == 1:
            data = data['Adj Close'].to_frame(tickers[0])
        else:
            data = data['Adj Close']

        mu = expected_returns.ema_historical_return(data)
        short_term_data = data.iloc[-30:]
        Sigma = risk_models.sample_cov(short_term_data)

        ef = EfficientFrontier(mu, Sigma)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        expected_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)

        st.subheader('Selected Ticker Data (Short-Term)')
        st.write(short_term_data)

        st.subheader('Optimized Portfolio Weights (Short-Term)')
        st.write(pd.Series(cleaned_weights))

        st.subheader('Efficient Frontier (Short-Term)')
        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(go.Scatter(x=np.sqrt(np.diag(Sigma)), y=mu, mode='markers', name=ticker))
        fig.add_trace(go.Scatter(x=[annual_volatility], y=[expected_return], mode='markers', marker=dict(size=15, color='red'), name='Optimized Portfolio (Short-Term)'))
        fig.update_layout(title='Efficient Frontier (Short-Term)', xaxis_title='Annual Volatility', yaxis_title='Expected Annual Return')
        st.plotly_chart(fig)

        st.subheader('Portfolio Metrics (Short-Term)')
        st.write(f'Expected Annual Return: {expected_return:.2%}')
        st.write(f'Annual Volatility: {annual_volatility:.2%}')
        st.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')

    except Exception as e:
        st.error(f"An error occurred: {e}")
