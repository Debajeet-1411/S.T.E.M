import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import wikipedia
from datetime import datetime
from main import update_stock_history, build_sentiment_dataset

st.set_page_config(page_title="Sentiment Stock Analyzer", layout="wide")
st.title("📈 Sentiment-Based Stock Movement Prediction")

# ───── Load and Prepare Data ─────────────────────────────────
with st.spinner("Loading stock and sentiment data..."):
    stock_df = update_stock_history()
    sentiment_df = build_sentiment_dataset(stock_df)
    model = joblib.load("sentiment_stock_model.joblib")

# ───── Sidebar: Ticker Selection ─────────────────────────────
tickers = sentiment_df["Ticker"].unique()
selected_ticker = st.sidebar.selectbox("Select a Stock Ticker", tickers)

# ───── Ticker Info Card from Wikipedia ───────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Ticker Info")
try:
    summary = wikipedia.summary(selected_ticker, sentences=2)
    st.sidebar.info(summary)
except:
    st.sidebar.warning("No summary found on Wikipedia.")

# ───── Date Range Filter ─────────────────────────────────────
min_date = pd.to_datetime(sentiment_df["Date"]).min()
max_date = pd.to_datetime(sentiment_df["Date"]).max()
date_range = st.sidebar.date_input("Filter by Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# ───── Filter Data for Selected Ticker ──────────────────────
df = sentiment_df[(sentiment_df["Ticker"] == selected_ticker)].copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df[(df["Date"] >= pd.to_datetime(date_range[0])) & (df["Date"] <= pd.to_datetime(date_range[1]))]

# ───── Show Candlestick Chart (Using Stock Data) ────────────
st.subheader(f"📊 Candlestick Chart for {selected_ticker}")
candlestick_df = stock_df[["Date", f"{selected_ticker}_Open", f"{selected_ticker}_High", f"{selected_ticker}_Low", f"{selected_ticker}_Close"]].dropna()
candlestick_df = candlestick_df.rename(columns={
    f"{selected_ticker}_Open": "Open",
    f"{selected_ticker}_High": "High",
    f"{selected_ticker}_Low": "Low",
    f"{selected_ticker}_Close": "Close"
})
candlestick_df = candlestick_df[(candlestick_df["Date"] >= pd.to_datetime(date_range[0])) & (candlestick_df["Date"] <= pd.to_datetime(date_range[1]))]

fig_candle = go.Figure(data=[
    go.Candlestick(
        x=candlestick_df["Date"],
        open=candlestick_df["Open"],
        high=candlestick_df["High"],
        low=candlestick_df["Low"],
        close=candlestick_df["Close"],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Price'
    )
])
fig_candle.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_white",
    height=500
)
st.plotly_chart(fig_candle, use_container_width=True)

# ───── Sentiment vs Stock Direction ─────────────────────────
st.subheader("📉 Sentiment Score vs Stock Movement")
fig_sentiment = go.Figure()
fig_sentiment.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Sentiment"],
    mode="lines+markers",
    name="Sentiment Score",
    line=dict(color="blue")
))
fig_sentiment.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Direction"],
    mode="lines+markers",
    name="Stock Movement (1 = Up, 0 = Down)",
    yaxis="y2",
    line=dict(color="green", dash="dot")
))
fig_sentiment.update_layout(
    xaxis=dict(title="Date"),
    yaxis=dict(title="Sentiment", side="left"),
    yaxis2=dict(title="Movement", overlaying="y", side="right", range=[-0.2, 1.2]),
    legend=dict(x=0.01, y=1),
    height=500,
    template="plotly_white"
)
st.plotly_chart(fig_sentiment, use_container_width=True)

st.markdown("""
> **ℹ️ Tip**: Sentiment score ranges from -1 (very negative) to 1 (very positive). 
> A direction of 1 indicates the stock closed higher than it opened.
""")

# ───── Trend Summary ────────────────────────────────────────
if len(df) > 2:
    trend_diff = df.iloc[-1]["Sentiment"] - df.iloc[0]["Sentiment"]
    trend = "🔺 Bullish" if trend_diff > 0 else "🔻 Bearish"
    st.info(f"Trend from {date_range[0]} to {date_range[1]}: {trend}")

# ───── Prediction from Custom Sentiment ─────────────────────
st.subheader("🔮 Predict Market Direction from Sentiment")
custom_sentiment = st.slider("Adjust Sentiment Score", -1.0, 1.0, 0.0, 0.01)
pred_proba = model.predict_proba([[custom_sentiment]])[0]
pred_label = model.predict([[custom_sentiment]])[0]

color = "green" if pred_label == 1 else "red"
label_text = "📈 Likely to Rise" if pred_label == 1 else "📉 Likely to Fall"

st.metric("Prediction", label_text)
st.progress(pred_proba[pred_label])

with st.expander("📖 How is this prediction made?"):
    st.write("The model uses past sentiment scores from news articles and Wikipedia summaries to learn patterns associated with stock movements. A logistic regression (or similar model) was trained to understand this relationship. Adjust the slider above to simulate different sentiment scenarios.")

# ───── Download Buttons ─────────────────────────────────────
st.download_button("📥 Download Sentiment Dataset", df.to_csv(index=False), file_name="sentiment_data.csv")

# ───── Footer ───────────────────────────────────────────────
st.caption("Made with ❤️ using Streamlit · Wikipedia · YFinance · VADER · by Debajeet Mandal")
