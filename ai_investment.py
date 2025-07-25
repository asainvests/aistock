import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from newsapi import NewsApiClient
from transformers import pipeline
import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# ----- 1. Stock Data Fetching and Prediction -----

def get_stock_data(ticker, interval='5m', period='1d'):
    """
    Fetches stock data for a given ticker symbol.
    """
    data = yf.download(tickers=ticker, interval=interval, period=period)
    return data

def predict_intraday(stock_df):
    """
    Predicts stock prices for the next few time points using linear regression.
    """
    stock_df = stock_df.copy()
    stock_df['returns'] = stock_df['Close'].pct_change()
    stock_df.dropna(inplace=True)

    # Prepare features (time indices) and target (stock close prices)
    X = np.arange(len(stock_df)).reshape(-1, 1)  # Time indices
    y = stock_df['Close'].values  # Close price

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Predict future prices (next 5 time points)
    future_index = np.array([[len(stock_df) + i] for i in range(1, 6)])
    future_index_scaled = scaler.transform(future_index)
    future_preds = model.predict(future_index_scaled)

    return future_preds, stock_df

def plot_predictions(stock_df, future_preds):
    """
    Plots the stock price and its predictions.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(stock_df['Close'], label='Past')
    plt.plot(range(len(stock_df), len(stock_df) + len(future_preds)), future_preds, label='Predicted', linestyle='--')
    plt.legend()
    plt.title("Intraday Price Forecast")
    plt.show()

# ----- 2. News API Integration and Sentiment Analysis -----

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='bcfd723986ed4a24a0d623175b7e2cd1')

def get_news_about(ticker):
    """
    Fetches news articles related to a stock ticker symbol.
    """
    news = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=5)
    return [article['title'] for article in news['articles']]

# Sentiment analysis using HuggingFace transformers
sentiment_pipeline = pipeline("sentiment-analysis", model="bert-base-uncased")

def analyze_sentiment(news_list):
    """
    Analyzes sentiment of a list of news articles.
    """
    return [sentiment_pipeline(article)[0] for article in news_list]

# ----- 3. Learning from Mistakes (Error Tracking and Feedback Loop) -----

def log_prediction(ticker, prediction, actual):
    """
    Logs predictions and their corresponding errors for future learning.
    """
    record = {
        'ticker': ticker,
        'timestamp': str(datetime.now()),
        'predicted': prediction.tolist(),
        'actual': actual.tolist(),
        'error': np.abs(actual - prediction) / np.maximum(actual, 1)
    }
    
    with open('prediction_log.json', 'a') as f:
        f.write(json.dumps(record) + '\n')

def retrain_model_with_feedback():
    """
    Retrieves logs and prepares for model retraining (if applicable).
    """
    with open('prediction_log.json', 'r') as f:
        logs = f.readlines()
    
    past_data = pd.read_json(logs)
    return past_data

# ----- 4. Advanced Predictive Model (LSTM) -----

class LSTMModel(nn.Module):
    """
    A simple LSTM model for stock price prediction.
    """
    def __init__(self, input_size=1, hidden_layer_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(stock_df, epochs=100):
    """
    Trains an LSTM model for stock price prediction.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(stock_df[['Close']].values)

    # Prepare data for LSTM (sequence of past prices)
    sequence_length = 60
    x_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Initialize and train LSTM model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model, scaler

# ----- 5. Interactive Web Interface (Streamlit) -----

def app():
    """
    Streamlit app that ties all components together.
    """
    st.title("AI-Powered Stock Prediction")

    ticker = st.text_input("Enter stock symbol", "AAPL")
    if st.button("Predict"):
        # Fetch stock data and make predictions
        data = get_stock_data(ticker)
        future_preds, stock_df = predict_intraday(data)
        plot_predictions(stock_df, future_preds)

        # Get related news and perform sentiment analysis
        news = get_news_about(ticker)
        st.write("🗞️ News:")
        for article in news:
            st.write(f"- {article}")
        
        sentiments = analyze_sentiment(news)
        st.write("🧠 Sentiment Analysis:")
        for sentiment in sentiments:
            st.write(f"- {sentiment['label']}: {sentiment['score']}")

        # Log prediction (for feedback loop)
        log_prediction(ticker, future_preds, stock_df['Close'].iloc[-1])

        # Optionally retrain model with feedback (advanced)
        if st.checkbox('Retrain model'):
            retrain_data = retrain_model_with_feedback()
            st.write("Retraining model with past feedback data...")

if __name__ == "__main__":
    app()
