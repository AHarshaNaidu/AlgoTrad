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

                st.write(f'## Cumulative Return for {ticker}')
                st.line_chart(df['cum_return'])

                st.write(f'## Bollinger Bands for {ticker}')
                fig = plot_bollinger_bands(df, ticker)
                st.plotly_chart(fig)

                st.write(df.tail())
            else:
                st.warning(f'No data found for {ticker}')
