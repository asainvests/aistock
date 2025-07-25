# 🚀 QuantumVest — Adaptive AI Stock Forecaster (with News Sentiment)

import yfinance as yf
import numpy as np
import pandas as pd
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
    if start is None:
        start = today - timedelta(days=90)  # get 3 months of data by default
    if isinstance(start, str):
        start = datetime.strptime(start, '%Y-%m-%d')
    if end is None:
        end = today
    if isinstance(end, str):
        end = datetime.strptime(end, '%Y-%m-%d')

    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'),
                     progress=False, interval="1d", auto_adjust=True)
    if df.empty:
        raise ValueError(f"⚠️ No daily data found for ticker '{ticker}'. Market may be closed or ticker is invalid.")
    df.dropna(inplace=True)
    return df

# ========== 2. News Sentiment Handling ==========

newsapi = NewsApiClient(api_key='bcfd723986ed4a24a0d623175b7e2cd1')

try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    sentiment_pipeline = None
    print("Warning: Sentiment model loading failed:", e)

def get_daily_sentiment(ticker, start_date, end_date):
    """Fetch news for each day and compute average sentiment score."""
    current_date = start_date
    daily_sentiments = {}

    while current_date <= end_date:
        try:
            articles = newsapi.get_everything(
                q=ticker,
                from_param=current_date.strftime('%Y-%m-%d'),
                to=current_date.strftime('%Y-%m-%d'),
                language='en',
                page_size=100,
                sort_by='relevancy'
            )
            texts = [a['title'] + ". " + (a['description'] or "") for a in articles.get('articles', [])]
            if texts and sentiment_pipeline:
                sentiments = sentiment_pipeline(texts)
                scores = []
                for s in sentiments:
                    if s['label'] == 'POSITIVE':
                        scores.append(s['score'])
                    elif s['label'] == 'NEGATIVE':
                        scores.append(-s['score'])
                    else:
                        scores.append(0)
                daily_sentiments[current_date] = np.mean(scores) if scores else 0.0
            else:
                daily_sentiments[current_date] = 0.0
        except Exception:
            daily_sentiments[current_date] = 0.0
        current_date += timedelta(days=1)

    sentiment_df = pd.DataFrame.from_dict(daily_sentiments, orient='index', columns=['Sentiment'])
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    return sentiment_df

def get_stock_and_sentiment_data(ticker, start_date, end_date):
    stock_df = get_stock_data(ticker, start_date, end_date)
    sentiment_df = get_daily_sentiment(ticker, stock_df.index.min(), stock_df.index.max())

    combined_df = stock_df.join(sentiment_df, how='left')
    combined_df['Sentiment'].fillna(method='ffill', inplace=True)
    combined_df['Sentiment'].fillna(0, inplace=True)

    return combined_df[['Close', 'Sentiment']]

# ========== 3. LSTM Deep Learning Model ==========

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, num_layers=4, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def prepare_sequences_multi_feature(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])
        y.append(data[i, 0])  # predict only 'Close' price
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

def train_lstm_model_multi(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    seq_len = 60
    x, y = prepare_sequences_multi_feature(scaled, seq_len)
    if len(x) == 0:
        raise ValueError("Not enough data to train the model.")

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    errors = load_error_history()
    epochs, lr = adapt_learning_parameters(errors)

    model = LSTMModel(input_size=2, hidden_size=256, num_layers=4)
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

def predict_future_multi(model, recent_data, scaler, future_days=5):
    predictions = []
    input_seq = recent_data[-60:]  # shape (60, features)

    for _ in range(future_days):
        input_reshaped = torch.tensor(input_seq.reshape(1, 60, input_seq.shape[1]), dtype=torch.float32)
        with torch.no_grad():
            pred = model(input_reshaped).item()

        # Append predicted price + dummy sentiment (0) for next day
        next_input = np.array([pred, 0.0])
        input_seq = np.vstack([input_seq[1:], next_input])
        predictions.append(pred)

    # Inverse transform only the Close price, ignore sentiment column for inverse scaling
    dummy_sentiment = np.zeros((future_days, 1))
    scaled_preds = np.hstack([np.array(predictions).reshape(-1,1), dummy_sentiment])
    inv_scaled = scaler.inverse_transform(scaled_preds)
    return inv_scaled[:, 0]

# ========== 4. Logging Predictions ==========

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

# ========== 5. Streamlit App UI ==========

def app():
    st.set_page_config(page_title="QuantumVest AI Forecaster", layout="wide")
    st.title("💼 QuantumVest — Institutional AI Stock Forecasting")
    st.markdown("Adaptive predictive engine trained on market memory and news sentiment. Adaptive, explainable, and accurate.")

    ticker = st.sidebar.text_input("📌 Ticker Symbol", "AAPL")
    start_date = st.sidebar.date_input("🗕️ Start Date", datetime.today() - timedelta(days=90))
    future_days = st.sidebar.slider("📉 Days to Forecast Ahead", 1, 10, 5)

    if st.sidebar.button("🚀 Forecast Now"):
        try:
            with st.spinner("Training neural network, decoding news cycles, adapting weights..."):
                df = get_stock_and_sentiment_data(ticker, start_date, datetime.today())

                model, scaler = train_lstm_model_multi(df)
                scaled_data = scaler.transform(df)
                preds = predict_future_multi(model, scaled_data, scaler, future_days)

                actuals = df['Close'].values[-future_days:] if len(df) >= future_days else [df['Close'].values[-1]]
                log_prediction(ticker, preds, actuals)

                st.subheader("📈 Forecast vs. History")
                last_price = df['Close'].iloc[-1]
                forecast_start = df.index[-1] + timedelta(days=1)
                forecast_dates = pd.date_range(start=forecast_start, periods=future_days, freq='B')
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
                news = []
                try:
                    articles = newsapi.get_everything(q=ticker, language='en', page_size=5, sort_by='publishedAt')
                    news = [a['title'] + ": " + (a['description'] or "") for a in articles.get('articles', [])]
                except:
                    pass

                sentiments = []
                for article in news:
                    try:
                        if sentiment_pipeline:
                            sentiments.append(sentiment_pipeline(article)[0])
                        else:
                            sentiments.append({'label': 'UNKNOWN', 'score': 0})
                    except:
                        sentiments.append({'label': 'UNKNOWN', 'score': 0})

                for article, sentiment in zip(news, sentiments):
                    st.markdown(f"- {article}")
                    st.markdown(f"  > Sentiment: **{sentiment['label']}** (Confidence: {sentiment['score']:.2f})")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app()
