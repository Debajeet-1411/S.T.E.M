import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from main import update_stock_history, build_sentiment_dataset

st.set_page_config(page_title="Sentiment Stock Analyzer", layout="wide")
st.title("📈 Sentiment-Based Stock Movement Prediction")

# Load data
with st.spinner("Loading stock data..."):
    stock_df = update_stock_history()
    sentiment_df = build_sentiment_dataset(stock_df)
    model = joblib.load("sentiment_stock_model.joblib")

# Sidebar - Filter
tickers = sentiment_df["Ticker"].unique()
selected_ticker = st.sidebar.selectbox("Select a Ticker", tickers)

# Filter data for selected ticker
df = sentiment_df[sentiment_df["Ticker"] == selected_ticker]

# Plotting stock data
st.subheader(f"📊 Historical Sentiment vs. Stock Direction for {selected_ticker}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pd.to_datetime(df["Date"]), df["Sentiment"], label="Sentiment", color="blue")
ax.set_ylabel("Sentiment Score")
ax2 = ax.twinx()
ax2.plot(pd.to_datetime(df["Date"]), df["Direction"], label="Stock Direction", color="green", alpha=0.6)
ax2.set_ylabel("Direction (1 = Up, 0 = Down)")
fig.legend(loc="upper right")
st.pyplot(fig)

# Make prediction
st.subheader("🔮 Predict Using Custom Sentiment Score")
custom_sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.01)
prediction = model.predict([[custom_sentiment]])
label = "📈 Rise" if prediction[0] == 1 else "📉 Fall"
st.metric("Model Prediction", label)

st.caption("Made with ❤️ for GSoC project exploration")
