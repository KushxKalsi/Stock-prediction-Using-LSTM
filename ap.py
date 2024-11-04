# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objs as go

xgb_model = joblib.load('xgboost_stock_model.pkl')
lstm_model = load_model('lstm_stock_model.h5')
scaler = joblib.load('scaler.pkl')

# Add custom CSS for a darker blue background and improved aesthetics
st.markdown("""
    <style>
        /* Set the background color */
        .main {
            background-color: #B3E5FC;
            padding: 20px;
            border-radius: 10px;
        }
        .stApp {
            background-color: #B3E5FC;
        }
        /* Customize the title */
        h1 {
            color: #01579B;
            text-align: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Style the selectbox */
        .stSelectbox {
            font-size: 16px;
            color: #004D40;
        }
        
        /* Customize other elements */
        .stButton>button {
            background-color: #0288D1;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 15px;
            cursor: pointer;
        }
        
        .stButton>button:hover {
            background-color: #0277BD;
        }
        
        /* Style the text */
        .stMarkdown, .stText {
            color: #004D40;
            font-family: 'Arial', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Function to fetch all stock tickers (Example: S&P 500)
def get_all_stock_tickers():
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    return sp500_tickers

# Function to download stock data
def download_stock_data(ticker):
    return yf.download(ticker, start="2018-01-01", end=datetime.now().strftime('%Y-%m-%d'))

# Preprocess the stock data and add technical indicators
def preprocess_data(df):
    # Add technical indicators
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

    # Drop any remaining NA values
    df.dropna(inplace=True)
    return df

# Function to predict stock movement for the next 5 days
def predict_stock(ticker):
    # Download and preprocess data
    stock_data = download_stock_data(ticker)
    processed_data = preprocess_data(stock_data)

    # Prepare input for the models
    X = processed_data[-50:]  # Last 50 days of data for prediction
    X = X[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Daily_Return', 'Log_Return',
           'SMA_50', 'EMA_20', 'RSI_14', 'Middle_Band', 'Upper_Band', 'Lower_Band',
           'Momentum', 'Volatility']].values  # Select required columns

    # Generate future dates starting from today (next 5 business days)
    future_dates = pd.date_range(datetime.now(), periods=6, freq='B')[1:]  # Get business days starting tomorrow

    # Check if we have enough data for LSTM
    if len(X) < 50:
        return None, None, None  # Not enough data

    # Make predictions with XGBoost
    xgb_pred = xgb_model.predict(X)

    # Prepare data for LSTM
    X_lstm = np.array([X])  # Reshape for LSTM model
    lstm_pred = lstm_model.predict(X_lstm)

    # Ensemble the predictions
    ensemble_pred = (xgb_pred + lstm_pred.flatten()) / 2

    # Determine the percentage change from previous price
    percentage_change = np.diff(ensemble_pred) / ensemble_pred[:-1] * 100
    return percentage_change, future_dates, processed_data

# Function to plot stock prices and predictions (Last 1 month + 5 days forecast)
def plot_stock_data(processed_data, future_dates, predictions):
    # Extract only the last 1 month of data (30 days) from the historical data
    one_month_data = processed_data.tail(30)

    fig = go.Figure()

    # Add historical stock prices for the last 1 month
    fig.add_trace(go.Scatter(x=one_month_data.index, y=one_month_data['Close'],
                             mode='lines', name='Last 1 Month Prices'))

    # Add predicted future prices
    future_prices = [one_month_data['Close'].values[-1]]  # Start with the last close price
    for pred in predictions:
        future_prices.append(future_prices[-1] * (1 + pred / 100))  # Apply percentage change

    # Plot the predicted future prices
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices[1:], mode='lines', name='Predicted Prices', line=dict(dash='dash')))

    # Customize the layout
    fig.update_layout(
        title='Stock Price Prediction (Last 1 Month + Next 5 Days)',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        template='plotly_dark'
    )

    st.plotly_chart(fig)

# Streamlit App UI
st.title("Stock Price Prediction (Next 5 Days)")

# Get all stock tickers (S&P 500 tickers here, but you can extend this)
all_stocks = get_all_stock_tickers()

# Display selectbox with all stock options
selected_stock = st.selectbox("Select a stock:", all_stocks)

if selected_stock:
    prediction, future_dates, processed_data = predict_stock(selected_stock)

    if prediction is not None and future_dates is not None:
        st.write(f"Prediction for {selected_stock}:")
        for date, pred in zip(future_dates, prediction):  # Display the percentage change for the next 5 days with dates
            st.write(f"On {date.date()}: **{pred:.2f}%** change in stock price.")

        # Plot the stock prices (Last 1 month + future 5-day predictions)
        plot_stock_data(processed_data, future_dates, prediction)
    else:
        st.write("Not enough data to make a prediction.")
