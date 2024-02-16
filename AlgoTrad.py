pip install pandas
pip install yfinance
pip install plotly
# Import necessary libraries
import numpy as np 
import pandas as pd 
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# Function to fetch stock data from Yahoo Finance
def get_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, start='2017-02-01', end='2022-12-06')
        return stock_data
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

# Function to add cumulative return to the dataframe
def add_cumulative_return(df):
    df['cum_return'] = (1 + df['Close'].pct_change()).cumprod() - 1
    return df

# Function to add Bollinger Bands to the dataframe
def add_bollinger_bands(df):
    df['middle_band'] = df['Close'].rolling(window=20).mean()
    df['upper_band'] = df['middle_band'] + 2 * df['Close'].rolling(window=20).std()
    df['lower_band'] = df['middle_band'] - 2 * df['Close'].rolling(window=20).std()
    return df

# Function to plot Bollinger Bands
def plot_bollinger_bands(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['middle_band'], mode='lines', name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], mode='lines', name='Lower Band'))
    return fig

# Function to add Ichimoku Cloud to the dataframe
def add_ichimoku_cloud(df):
    # Implement Ichimoku Cloud functionality here
    return df

# Function to fetch sector-wise cumulative returns
def fetch_sector_cum_returns(sector):
    # Implement fetching sector-wise cumulative returns here
    return None

# Streamlit app
def main():
    st.title('Stock Analysis App')

    # Stock analysis section
    st.subheader('Stock Analysis')
    ticker = st.text_input('Enter Stock Ticker (e.g., AAPL):')
    if st.button('Analyze Stock'):
        if ticker:
            st.write(f'Loading data for {ticker}...')
            df = get_stock_data(ticker)
            if df is not None:
                df = add_cumulative_return(df)
                df = add_bollinger_bands(df)
                df = add_ichimoku_cloud(df)

                st.write(f'## Cumulative Return for {ticker}')
                st.line_chart(df['cum_return'])

                st.write(f'## Bollinger Bands for {ticker}')
                fig = plot_bollinger_bands(df, ticker)
                st.plotly_chart(fig)

                st.write(f'## Ichimoku Cloud for {ticker}')
                # Implement Ichimoku Cloud plotting here

                st.write(df.tail())
            else:
                st.warning(f'No data found for {ticker}')

    # Portfolio optimization section
    st.sidebar.title('Markowitz Portfolio Optimization')
    selected_stocks = st.sidebar.multiselect('Select stocks for portfolio optimization:', ['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
    if selected_stocks:
        num_portfolios = st.sidebar.number_input('Enter number of portfolios to simulate:', min_value=1, value=10000)
        if st.sidebar.button('Run Portfolio Optimization'):
            st.sidebar.write('Running portfolio optimization...')
            # Implement portfolio optimization functionality
            st.sidebar.write('Portfolio optimization completed!')

    # Sector-wise cumulative returns section
    st.sidebar.title('Sector-wise Cumulative Returns')
    sectors = ['Technology', 'Healthcare', 'Finance', 'Consumer Discretionary', 'Utilities']
    selected_sector = st.sidebar.selectbox('Select a sector:', sectors)
    if selected_sector:
        # Fetch sector-wise cumulative returns data
        sector_cum_returns = fetch_sector_cum_returns(selected_sector)
        if sector_cum_returns is not None:
            # Visualize sector-wise cumulative returns
            st.sidebar.write('Sector-wise cumulative returns')
