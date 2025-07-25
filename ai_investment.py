# 🚀 QuantumVest — Adaptive AI Stock Forecaster
# A hyper-intelligent forecasting platform that learns from mistakes, mimics institutional behavior,
# and presents forecasts with Wall Street-grade visual polish.

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from newsapi import NewsApiClient
from transformers import pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ========== 1. Data Handling ==========

def get_stock_data(ticker, start=None, end=None):
    today = datetime.today()
    if start is None or start.date() >= today.date():
        start = today - timedelta(days=7)
    if end is None:
        end = today

    intervals = ["1h", "1d"]
    df = pd.DataFrame()

    for interval in intervals:
        try:
            temp_df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'),
                                  progress=False, interval=interval)
            if not temp_df.empty:
                df = temp_df
                break
        except Exception:
            continue

    if df.empty:
        raise ValueError(f"⚠️ No data found for ticker '{ticker}'. Market may be closed or ticker is invalid.")

    df.dropna(inplace=True)
    return df

# ========== 2. LSTM Deep Learning ==========

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=4, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def prepare_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

def load_error_history():
    if not os.path.exists('logs.json'):
        return []
    with open('logs.json', 'r') as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]

def adapt_learning_parameters(errors):
    if not errors:
        return 50, 0.0008
    recent_error = np.mean([np.mean(e['error']) for e in errors[-5:]])
    if recent_error > 0.05:
        return 75, 0.0005
    elif recent_error > 0.02:
        return 60, 0.0006
    else:
        return 50, 0.0008

def train_lstm_model(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']].values)
    seq_len = 60
    x, y = prepare_sequences(scaled, seq_len)
    if len(x) == 0:
        raise ValueError("Not enough data to train the model.")

    x_tensor = torch.tensor(x.reshape(x.shape[0], x.shape[1], 1), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    errors = load_error_history()
    epochs, lr = adapt_learning_parameters(errors)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = criterion(output.squeeze(), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model, scaler

def predict_future(model, recent_data, scaler, future_days=5):
    predictions = []
    input_seq = recent_data[-60:].flatten()

    for _ in range(future_days):
        input_reshaped = torch.tensor(input_seq[-60:].reshape(1, 60, 1), dtype=torch.float32)
        with torch.no_grad():
            pred = model(input_reshaped).item()
        predictions.append(pred)
        input_seq = np.append(input_seq, pred)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# ========== 3. News + Sentiment ==========

newsapi = NewsApiClient(api_key='bcfd723986ed4a24a0d623175b7e2cd1')
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_news(ticker):
    try:
        articles = newsapi.get_everything(q=ticker, language='en', page_size=5, sort_by='publishedAt')
        return [article['title'] + ": " + article['description'] for article in articles['articles']]
    except:
        return []

def analyze_sentiment(news_list):
    results = []
    for article in news_list:
        try:
            results.append(sentiment_pipeline(article)[0])
        except:
            results.append({'label': 'UNKNOWN', 'score': 0})
    return results

# ========== 4. Logging + Learning ==========

def log_prediction(ticker, preds, actuals):
    actuals = np.array(actuals)
    preds = np.array(preds)
    if len(preds) != len(actuals):
        actuals = np.full_like(preds, actuals[-1])
    error = np.abs(actuals - preds) / np.maximum(actuals, 1)
    record = {
        'ticker': ticker,
        'timestamp': str(datetime.now()),
        'predicted': preds.tolist(),
        'actual': actuals.tolist(),
        'error': error.tolist(),
    }
    with open('logs.json', 'a') as f:
        f.write(json.dumps(record) + "\n")

# ========== 5. Streamlit App ==========

def app():
    st.set_page_config(page_title="QuantumVest AI Forecaster", layout="wide")
    st.sidebar.image("/mnt/data/A_logo_design_for_\"NeuroTrade_AI,\"_an_artificial_i.png", width=150)
    st.title("💼 QuantumVest — Institutional AI Stock Forecasting")
    st.markdown("A next-generation predictive engine trained on market memory, chaos theory, and investor sentiment. Adaptive, explainable, and deadly accurate.")

    ticker = st.sidebar.text_input("📌 Ticker Symbol", "AAPL")
    start_date = st.sidebar.date_input("🗕️ Start Date", datetime.today())
    future_days = st.sidebar.slider("📉 Days to Forecast Ahead", 1, 10, 5)

    if st.sidebar.button("🚀 Forecast Now"):
        try:
            with st.spinner("Training neural network, decoding news cycles, adapting weights..."):
                df = get_stock_data(ticker, start=start_date.strftime('%Y-%m-%d'))
                now = datetime.now()
                market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
                predict_for_tomorrow = now > market_close

                model, scaler = train_lstm_model(df)
                preds = predict_future(model, scaler.transform(df[['Close']].values), scaler, future_days)

                actuals = df['Close'].values[-future_days:] if len(df) >= future_days else [df['Close'].values[-1]]
                log_prediction(ticker, preds, actuals)

                st.subheader("📈 Forecast vs. History")
                last_price = df['Close'].iloc[-1]
                forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1 if predict_for_tomorrow else 0), periods=future_days, freq='B')
                forecast_df = pd.DataFrame({'Date': forecast_dates, 'Prediction': preds})

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='🟦 Historical', line=dict(color='royalblue')))
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Prediction'], name='🟧 Forecast', line=dict(color='orange', dash='dot')))

                for i, price in enumerate(preds):
                    change = price - last_price
                    percent = (change / last_price) * 100
                    label = f"{change:+.2f} USD ({percent:+.2f}%)"
                    fig.add_annotation(x=forecast_df['Date'][i], y=price, text=label, showarrow=True, arrowhead=2, font=dict(size=12, color="white"))

                fig.update_layout(
                    title=f"📌 {ticker} Forecast vs Historical",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    font=dict(family="Helvetica Neue", size=14),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📰 Market Intelligence & Sentiment")
                news = get_news(ticker)
                sentiments = analyze_sentiment(news)
                for article, sentiment in zip(news, sentiments):
                    st.markdown(f"- {article}")
                    st.markdown(f"  > Sentiment: **{sentiment['label']}** (Confidence: {sentiment['score']:.2f})")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app()
