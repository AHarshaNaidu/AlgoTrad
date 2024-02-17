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
