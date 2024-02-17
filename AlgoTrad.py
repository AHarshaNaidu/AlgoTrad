import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
from ta.trend import IchimokuIndicator
import plotly.graph_objs as go

# App title
st.markdown('''
# Stock Price App
Shown are the stock price data for query companies!
''')
st.write('---')

# Sidebar
st.sidebar.subheader('Query parameters')

# Manually enter the stock exchange
stock_exchange = st.sidebar.text_input("Enter Stock Exchange (e.g., NASDAQ, NYSE)", '')

# Manually enter the list of stocks separated by commas
stocks_input = st.sidebar.text_input("Enter Stock Symbols (e.g., AAPL, MSFT, GOOGL)", '')

# Convert the list of stocks to uppercase and split by comma
stocks = [stock.strip().upper() for stock in stocks_input.split(',') if stock.strip()]

if stocks:
    # Retrieve tickers data
    tickerData = yf.Tickers(" ".join(stocks))  # Concatenate the list of stocks into a space-separated string
    tickerDf = tickerData.history(period='1d', start=datetime.date(2019, 1, 1), end=datetime.date(2021, 1, 31))

    # Ticker information
    st.header('**Ticker information**')
    for stock in stocks:
        string_logo = ''
        if stock_exchange and stock:
            ticker = yf.Ticker(f"{stock}.{stock_exchange}")
            if 'logo_url' in ticker.info:
                string_logo = '<img src=%s>' % ticker.info['logo_url']
                st.markdown(string_logo, unsafe_allow_html=True)

            string_name = ticker.info.get('longName', 'N/A')
            st.subheader(f"{stock} - {string_name}")
        else:
            st.subheader("Please enter both stock exchange and stock symbols")

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
                                   go.Scatter(x=tickerDf.index, y=tickerDf['ichimoku_conversion_line'],
                                              name='Conversion Line')],
                             layout=go.Layout(title='Ichimoku Cloud'))
    st.plotly_chart(fig_ichimoku)
else:
    st.warning("Please enter at least one stock symbol.")
