import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import cufflinks as cf
import plotly.express as px
import datetime

# Function to fetch list of stocks from a given stock exchange
def get_stock_list(stock_exchange):
    # Logic to fetch the list of stocks from the selected stock exchange
    # Replace this with your code to fetch the list of stocks
    return ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"]  # Example list of stocks

# Function to merge dataframes by column name
def merge_df_by_column_name(column_name, start_date, end_date, *tickers):
    dfs = []
    for ticker in tickers:
        ticker_data = yf.download(ticker, start=start_date, end=end_date)
        dfs.append(ticker_data[[column_name]].rename(columns={column_name: ticker}))
    merged_df = pd.concat(dfs, axis=1)
    return merged_df

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Stock Price", "Portfolio Optimization"])

if page == "Stock Price":
    # App title and description
    st.markdown('''
    # Stock Price App
    Shown are the stock price data for query companies!
    ''')
    st.write('---')

    # Sidebar parameters
    st.sidebar.subheader('Query parameters')
    start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
    end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

    # Stock exchange selection
    stock_exchange = st.sidebar.selectbox("Select Stock Exchange", ["NASDAQ", "NYSE", "London Stock Exchange", "Tokyo Stock Exchange"])

    # Fetch list of stocks from the selected stock exchange
    stock_list = get_stock_list(stock_exchange)

    # Allow user to select a stock
    tickerSymbol = st.sidebar.selectbox('Stock ticker', stock_list)  # Select ticker symbol
    tickerData = yf.Ticker(tickerSymbol)  # Get ticker data
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)  # Get historical prices for this ticker

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
    qf = cf.QuantFig(tickerDf, title='Stock Price Analysis', legend='top', name='Stock Price')
    qf.add_bollinger_bands()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

elif page == "Portfolio Optimization":
    # App title and description
    st.markdown('''
    # Markowitz Portfolio Optimization
    Harry Markowitz proved that you could make what is called an efficient portfolio. That is a portfolio that optimizes return while also minimizing risk. We don't benefit from analyzing individual securities at the same rate as if we instead considered a portfolio of stocks.

    We do this by creating portfolios with stocks that are not correlated. We want to calculate expected returns by analyzing the returns of each stock multiplied by its weight.

    The standard deviation of the portfolio is found this way. Sum multiple calculations starting by finding the product of the first securities weight squared times its standard deviation squared. The middle is 2 times the correlation coefficient between the stocks. And, finally add those to the weight squared times the standard deviation squared for the second security.

    Plotting an Efficient Frontier
    ''')

    # Stock exchange selection for portfolio optimization
    stock_exchange = st.selectbox("Select Stock Exchange for Portfolio Optimization", ["NASDAQ", "NYSE", "London Stock Exchange", "Tokyo Stock Exchange"])

    # Fetch list of stocks from the selected stock exchange for portfolio optimization
    stock_list = get_stock_list(stock_exchange)

    # List of stocks for portfolio optimization
    port_list = st.multiselect("Select Stocks for Portfolio Optimization", stock_list)
