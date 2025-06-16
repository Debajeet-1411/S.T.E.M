import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import wikipedia
from datetime import datetime, timedelta
from main import update_stock_history, build_sentiment_dataset

st.set_page_config(page_title="Sentiment Stock Analyzer", layout="wide")
st.title("📈 Sentiment‑Based Stock Movement Prediction")

# ───── Mapping for full company names ───────────────────────
TICKER_MAP = {
    "TSLA": "Tesla, Inc.",
    "GOOGL": "Alphabet Inc.",
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon (company)",
    "MSFT": "Microsoft Corporation"
}

# ───── Load and prepare data ────────────────────────────────
with st.spinner("Loading data & model…"):
    stock_df = update_stock_history()
    sentiment_df = build_sentiment_dataset(stock_df)
    model = joblib.load("sentiment_stock_model.joblib")

# ───── Sidebar: stock selector with full names ──────────────
choices = [f"{TICKER_MAP[t]} ({t})" for t in TICKER_MAP]
selection = st.sidebar.selectbox("Select a Stock", choices)
selected_ticker = selection.split("(")[-1].rstrip(")")  # Extract ticker
full_name = TICKER_MAP[selected_ticker]

# ───── Company info card ────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🏷️ Company Info")
try:
    summary = wikipedia.summary(full_name, sentences=2)
    st.sidebar.info(summary)
except Exception:
    st.sidebar.warning("No summary found on Wikipedia.")

# ───── Date range filter ────────────────────────────────────
min_date = pd.to_datetime(sentiment_df["Date"]).min()
max_date = pd.to_datetime(sentiment_df["Date"]).max()
start_date, end_date = st.sidebar.date_input(
    "Filter by Date Range",
    [min_date, max_date],
    min_value=min_date, max_value=max_date
)

# ───── Filter dataset ───────────────────────────────────────
df = sentiment_df[(sentiment_df["Ticker"] == selected_ticker)].copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

# ───── Candlestick chart with range slider ──────────────────
st.subheader(f"🕯 Candlestick Chart · {full_name} ({selected_ticker})")
cs = stock_df[["Date", f"{selected_ticker}_Open", f"{selected_ticker}_High", f"{selected_ticker}_Low", f"{selected_ticker}_Close"]].dropna()
cs.columns = ["Date", "Open", "High", "Low", "Close"]
cs = cs[(cs["Date"] >= pd.to_datetime(start_date)) & (cs["Date"] <= pd.to_datetime(end_date))]

fig_candle = go.Figure(go.Candlestick(
    x=cs["Date"], open=cs["Open"], high=cs["High"], low=cs["Low"], close=cs["Close"],
    increasing_line_color="green", decreasing_line_color="red"))
fig_candle.update_layout(
    template="plotly_white", height=450,
    xaxis_title="Date", yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=True)
st.plotly_chart(fig_candle, use_container_width=True)

# ───── Sentiment vs movement with range slider ──────────────
st.subheader("📉 Sentiment Score vs Movement")
fig_sent = go.Figure()
fig_sent.add_trace(go.Scatter(x=df["Date"], y=df["Sentiment"], mode="lines", name="Sentiment", line=dict(color="blue")))
fig_sent.add_trace(go.Scatter(x=df["Date"], y=df["Direction"], mode="lines", name="Direction (1=Up)", yaxis="y2", line=dict(color="green", dash="dot")))
fig_sent.update_layout(
    template="plotly_white", height=450,
    xaxis=dict(title="Date", rangeslider=dict(visible=True)),
    yaxis=dict(title="Sentiment"),
    yaxis2=dict(title="Movement", overlaying="y", side="right", range=[-0.2, 1.2]))
st.plotly_chart(fig_sent, use_container_width=True)

# ───── Trend summary ────────────────────────────────────────
if not df.empty:
    trend = "🔺 Bullish" if df["Sentiment"].iloc[-1] > df["Sentiment"].iloc[0] else "🔻 Bearish"
    st.info(f"Overall sentiment trend over selected period: {trend}")

# ───── Prediction slider & future projection ────────────────
st.subheader("🔮 Predict Direction & 10‑Day Outlook")
custom_sent = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.01)
prob = model.predict_proba([[custom_sent]])[0]
label = "Rise" if prob.argmax() == 1 else "Fall"
color = "green" if label == "Rise" else "red"

st.metric(label="Prediction", value=f"📈 {label}" if label=="Rise" else f"📉 {label}")
st.progress(prob.max())

# Future hourly prediction (simple demo using same probability)
future_hours = pd.date_range(datetime.now(), periods=10*24, freq="H")
outlook = pd.DataFrame({"Datetime": future_hours, "ProbRise": prob[1]})
fig_future = go.Figure(go.Scatter(x=outlook["Datetime"], y=outlook["ProbRise"], mode="lines", line=dict(color="purple")))
fig_future.update_layout(template="plotly_white", height=350, yaxis_title="Probability of Rise", xaxis_title="Future (10 days, hourly)")
st.plotly_chart(fig_future, use_container_width=True)

# ───── Download dataset ─────────────────────────────────────
st.download_button("📥 Download Filtered Dataset", df.to_csv(index=False), file_name=f"{selected_ticker}_sentiment.csv")

st.caption("Made with ❤️ using Streamlit · Wikipedia · YFinance · VADER · by Debajeet Mandal")
