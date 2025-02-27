import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime
import plotly.graph_objs as go

# Load models and scaler safely
try:
    xgb_model = joblib.load('xgboost_stock_model.pkl')
    lstm_model = load_model('lstm_stock_model.h5')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading models: {e}")

# Function to fetch stock tickers (Example: S&P 500)
@st.cache_data
def get_all_stock_tickers():
    try:
        sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        return sp500_tickers
    except Exception as e:
        st.warning(f"Could not fetch stock tickers: {e}")
        return ['AAPL', 'GOOGL', 'MSFT']  # Default fallback

# Function to download stock data using latest yfinance
def download_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")  # Last 5 years of data

        if df.empty:
            st.warning(f"No data found for {ticker}")
            return None

        # Ensure required columns exist
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df.set_index('Date', inplace=True)
        
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']  # Fallback if missing

        return df
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        return None

# Preprocess the stock data
def preprocess_data(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['Close'].rolling(window=20).std() * 2)
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    df['Volatility'] = df['Close'].rolling(window=21).std()

    df.dropna(inplace=True)
    return df

# Function to predict stock movement for the next 5 days
def predict_stock(ticker):
    stock_data = download_stock_data(ticker)
    if stock_data is None:
        return None, None, None

    processed_data = preprocess_data(stock_data)

    required_features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Daily_Return', 'Log_Return',
                         'SMA_50', 'EMA_20', 'RSI_14', 'Middle_Band', 'Upper_Band', 'Lower_Band',
                         'Momentum', 'Volatility']

    if not all(feature in processed_data.columns for feature in required_features):
        st.warning(f"Insufficient data for {ticker}. Skipping prediction.")
        return None, None, None

    X = processed_data[-50:][required_features].values

    if len(X) < 50:
        return None, None, None

    future_dates = pd.date_range(datetime.now(), periods=6, freq='B')[1:]

    xgb_pred = xgb_model.predict(X)
    lstm_pred = lstm_model.predict(np.array([X]))

    ensemble_pred = (xgb_pred + lstm_pred.flatten()) / 2
    percentage_change = np.diff(ensemble_pred) / ensemble_pred[:-1] * 100
    return percentage_change, future_dates, processed_data

# Function to plot stock prices and predictions
def plot_stock_data(processed_data, future_dates, predictions):
    one_month_data = processed_data.tail(365)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=one_month_data.index, y=one_month_data['Close'],
                             mode='lines', name='Last 1 Month Prices', line=dict(color='darkblue')))

    future_prices = [one_month_data['Close'].values[-1]]
    for pred in predictions:
        future_prices.append(future_prices[-1] * (1 + pred / 100))

    fig.add_trace(go.Scatter(x=future_dates, y=future_prices[1:], mode='lines', name='Predicted Prices',
                             line=dict(dash='dash', color='red')))

    fig.update_layout(
        title='Stock Price Prediction (Last 1 Month + Next 5 Days)',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )

    st.plotly_chart(fig)

# Streamlit UI
st.title("Stock Price Prediction (Next 5 Days)")

all_stocks = get_all_stock_tickers()

selected_stock = st.selectbox("Select a stock:", all_stocks)
manual_stock = st.text_input("Or enter a stock symbol:")

stock_symbol = manual_stock.upper() if manual_stock else selected_stock

if stock_symbol:
    prediction, future_dates, processed_data = predict_stock(stock_symbol)

    if prediction is not None and future_dates is not None:
        st.write(f"Prediction for {stock_symbol}:")
        for date, pred in zip(future_dates, prediction):
            st.write(f"On {date.date()}: **{pred:.2f}%** change in stock price.")

        plot_stock_data(processed_data, future_dates, prediction)
    else:
        st.write("Not enough data to make a prediction.")
