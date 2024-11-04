import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime
import plotly.graph_objs as go

# Load models and scaler
xgb_model = joblib.load('xgboost_stock_model.pkl')
lstm_model = load_model('lstm_stock_model.h5')
scaler = joblib.load('scaler.pkl')

# Function to fetch all stock tickers (Example: S&P 500)
def get_all_stock_tickers():
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    return sp500_tickers

# Function to download stock data
def download_stock_data(ticker):
    return yf.download(ticker, start="2018-01-01", end=datetime.now().strftime('%Y-%m-%d'))

# Preprocess the stock data and add technical indicators
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
    processed_data = preprocess_data(stock_data)

    X = processed_data[-50:]
    X = X[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Daily_Return', 'Log_Return',
           'SMA_50', 'EMA_20', 'RSI_14', 'Middle_Band', 'Upper_Band', 'Lower_Band',
           'Momentum', 'Volatility']].values

    future_dates = pd.date_range(datetime.now(), periods=6, freq='B')[1:]

    if len(X) < 50:
        return None, None, None

    xgb_pred = xgb_model.predict(X)

    X_lstm = np.array([X])
    lstm_pred = lstm_model.predict(X_lstm)

    ensemble_pred = (xgb_pred + lstm_pred.flatten()) / 2
    percentage_change = np.diff(ensemble_pred) / ensemble_pred[:-1] * 100
    return percentage_change, future_dates, processed_data

# Function to plot stock prices and predictions (Last 1 month + 5 days forecast)
def plot_stock_data(processed_data, future_dates, predictions):
    one_month_data = processed_data.tail(30)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=one_month_data.index, y=one_month_data['Close'],
                             mode='lines', name='Last 1 Month Prices', line=dict(color='darkblue')))

    future_prices = [one_month_data['Close'].values[-1]]
    for pred in predictions:
        future_prices.append(future_prices[-1] * (1 + pred / 100))

    fig.add_trace(go.Scatter(x=future_dates, y=future_prices[1:], mode='lines', name='Predicted Prices',
                             line=dict(dash='dash', color='red')))

    fig.update_layout(
        title=dict(text='Stock Price Prediction (Last 1 Month + Next 5 Days)', font=dict(color='black')),
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title=dict(text='Legend', font=dict(color='black')),
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="right",
            x=1,
            font=dict(color='black')
        )
    )

    st.plotly_chart(fig)

# Streamlit App UI
st.title("Stock Price Prediction (Next 5 Days)")

all_stocks = get_all_stock_tickers()

selected_stock = st.selectbox("Select a stock:", all_stocks)
manual_stock = st.text_input("Or enter a stock symbol:")

if manual_stock:
    stock_symbol = manual_stock.upper()
else:
    stock_symbol = selected_stock

if stock_symbol:
    prediction, future_dates, processed_data = predict_stock(stock_symbol)

    if prediction is not None and future_dates is not None:
        st.write(f"Prediction for {stock_symbol}:")
        for date, pred in zip(future_dates, prediction):
            st.write(f"On {date.date()}: **{pred:.2f}%** change in stock price.")

        plot_stock_data(processed_data, future_dates, prediction)
    else:
        st.write("Not enough data to make a prediction.")
