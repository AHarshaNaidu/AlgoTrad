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
import requests
import io
from os import listdir
from os.path import isfile, join

# Function to get stock file names in a list
def get_stock_file_names(path):
    files = [x for x in listdir(path) if isfile(join(path, x))]
    tickers = [x.split('.')[0] for x in files]
    return tickers

# Function to return a DataFrame from a CSV
def get_stock_df_from_csv(url):
    try:
        df = pd.read_csv(url, index_col=0)
    except FileNotFoundError as ex:
        print(ex)
    else:
        return df

# Function to merge multiple stocks into one DataFrame by column name
def merge_df_by_column_name(col_name, sdate, edate, *dfs):
    mult_df = pd.DataFrame()
    for df in dfs:
        mask = (df.index >= sdate) & (df.index <= edate)
        mult_df[df.columns[0]] = df.loc[mask][col_name]
    return mult_df

# Function for Markowitz portfolio optimization
# Function for Markowitz portfolio optimization
def markowitz_portfolio_optimization(returns, risk_free_rate):
    # Placeholder implementation
    returns_numeric = pd.DataFrame()  # Create an empty DataFrame to store numeric values
    for column in returns.columns:
    returns_numeric[column] = pd.to_numeric(returns[column], errors='coerce')  # Convert values to numeric, coerce errors to NaN
    returns_numeric = returns_numeric.dropna()  # Drop rows with NaN values
    max_sharpe_volatility = returns_numeric.std() * np.sqrt(252)
    max_sharpe_volatility = returns.std() * np.sqrt(252)
    max_sharpe_weight = np.ones(len(returns.columns)) / len(returns.columns)
    return max_sharpe_return, max_sharpe_volatility, max_sharpe_weight

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    pages = ["Stock Analysis", "Portfolio Optimization"]
    choice = st.sidebar.radio("Go to", pages)

    if choice == "Stock Analysis":
        st.title("Stock Analysis")

        # Sidebar for query parameters
        st.sidebar.header("Query Parameters")
        start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2019-01-01'))
        end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2021-01-31'))

        # Retrieving tickers data
        url = 'https://raw.githubusercontent.com/AHarshaNaidu/AlgoTrad/main/Bombay.csv'
        response = requests.get(url)
        if response.status_code == 200:
            ticker_list = pd.read_csv(io.StringIO(response.text))
            tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list['Ticker'])  # Select ticker symbol
            tickerData = yf.Ticker(tickerSymbol)  # Get ticker data
            tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)  # Get historical prices

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

                # Stock Price Prediction using LSTM
                st.header('**Stock Price Prediction using LSTM**')

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
        else:
            st.error("Failed to fetch ticker data from the provided URL.")

    elif choice == "Portfolio Optimization":
        st.title("Portfolio Optimization")

        # Retrieving portfolio data
        url_portfolio = 'https://raw.githubusercontent.com/AHarshaNaidu/AlgoTrad/main/Bombay.csv'
        portfolio_df = get_stock_df_from_csv(url_portfolio)

        # Check if the DataFrame is not empty
        if portfolio_df is not None:
            # Display portfolio data
            st.write(portfolio_df)

            # Portfolio optimization
            risk_free_rate_portfolio = 0.05  # Example risk-free rate
            max_sharpe_return_portfolio, max_sharpe_volatility_portfolio, max_sharpe_weight_portfolio = markowitz_portfolio_optimization(portfolio_df, risk_free_rate_portfolio)

            st.header("Portfolio Optimization Results")
            st.write("Max Sharpe Ratio Portfolio Return:", max_sharpe_return_portfolio)
            st.write("Max Sharpe Ratio Portfolio Volatility:", max_sharpe_volatility_portfolio)
            st.write("Max Sharpe Ratio Portfolio Weight:", max_sharpe_weight_portfolio)
        else:
            st.error("Failed to fetch portfolio data from the provided URL.")

# Run the app
if __name__ == "__main__":
    main()
