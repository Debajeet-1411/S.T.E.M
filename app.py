import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

# Local modules
from main import update_stock_history, build_sentiment_dataset

# ────────────────────────────────────────────────────────────────
# Streamlit page config
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sentiment Stock Analyzer", layout="wide")
st.title("📈 Sentiment‑Based Stock Movement Prediction")

# ────────────────────────────────────────────────────────────────
# Constants & mappings
# ────────────────────────────────────────────────────────────────
TICKER_MAP = {
    "TSLA": "Tesla, Inc.",
    "GOOGL": "Alphabet Inc.",
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon (company)",
    "MSFT": "Microsoft Corporation",
}
MODEL_PATH = Path("models/sentiment_stock_model.joblib")

# ────────────────────────────────────────────────────────────────
# Cached helpers (refresh every hour)
# ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_and_sentiment():
    """Download/refresh the stock history & sentiment dataset (hourly cache)."""
    stock_df = update_stock_history()
    sentiment_df = build_sentiment_dataset(stock_df)
    return stock_df, sentiment_df


@st.cache_data(show_spinner=False)
def get_company_summary(title: str) -> str:
    """Retrieve a short Wikipedia summary with autosuggest disabled."""
    import wikipedia  # local import keeps cold‑start fast

    try:
        return wikipedia.summary(title, sentences=2, auto_suggest=False)
    except Exception:
        return "No summary found."


# ────────────────────────────────────────────────────────────────
# Load data & model (with spinner)
# ────────────────────────────────────────────────────────────────
with st.spinner("Loading data & model…"):
    STOCK_DF, SENTIMENT_DF = load_stock_and_sentiment()
    MODEL = joblib.load(MODEL_PATH)

# ────────────────────────────────────────────────────────────────
# Sidebar: stock selector & company info
# ────────────────────────────────────────────────────────────────
choices = [f"{name} ({ticker})" for ticker, name in TICKER_MAP.items()]
selection = st.sidebar.selectbox("Select a Stock", choices)
SELECTED_TICKER = selection.split("(")[-1].rstrip(")")
FULL_NAME = TICKER_MAP[SELECTED_TICKER]

st.sidebar.markdown("---")
st.sidebar.subheader("🏷️ Company Info")
st.sidebar.info(get_company_summary(FULL_NAME))

# ────────────────────────────────────────────────────────────────
# Date range filter
# ────────────────────────────────────────────────────────────────
min_date = pd.to_datetime(SENTIMENT_DF["Date"]).min()
max_date = pd.to_datetime(SENTIMENT_DF["Date"]).max()
user_dates = st.sidebar.date_input(
    "Filter by Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date,
)
# Handle tuple vs single return
if isinstance(user_dates, (list, tuple)):
    start_date, end_date = user_dates
else:
    start_date = end_date = user_dates

# ────────────────────────────────────────────────────────────────
# Filter datasets for the selected ticker & date window
# ────────────────────────────────────────────────────────────────
sel_df = (
    SENTIMENT_DF[SENTIMENT_DF["Ticker"] == SELECTED_TICKER].copy()
)
sel_df["Date"] = pd.to_datetime(sel_df["Date"])
sel_df = sel_df[(sel_df["Date"] >= pd.to_datetime(start_date)) & (sel_df["Date"] <= pd.to_datetime(end_date))]

cs_df = STOCK_DF[[
    "Date",
    f"{SELECTED_TICKER}_Open",
    f"{SELECTED_TICKER}_High",
    f"{SELECTED_TICKER}_Low",
    f"{SELECTED_TICKER}_Close",
]].dropna()
cs_df.columns = ["Date", "Open", "High", "Low", "Close"]
cs_df = cs_df[(cs_df["Date"] >= pd.to_datetime(start_date)) & (cs_df["Date"] <= pd.to_datetime(end_date))]

# ────────────────────────────────────────────────────────────────
# Candlestick chart
# ────────────────────────────────────────────────────────────────
st.subheader(f"🕯 Candlestick Chart · {FULL_NAME} ({SELECTED_TICKER})")
fig_candle = go.Figure(
    go.Candlestick(
        x=cs_df["Date"],
        open=cs_df["Open"],
        high=cs_df["High"],
        low=cs_df["Low"],
        close=cs_df["Close"],
        increasing_line_color="green",
        decreasing_line_color="red",
    )
)
fig_candle.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
)
fig_candle.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig_candle, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# Sentiment vs next‑day direction chart
# ────────────────────────────────────────────────────────────────
st.subheader("📉 Sentiment Score vs Movement")
fig_sent = go.Figure()
fig_sent.add_trace(
    go.Scatter(
        x=sel_df["Date"],
        y=sel_df["Sentiment"],
        mode="lines",
        name="Sentiment",
        line=dict(color="blue"),
    )
)
fig_sent.add_trace(
    go.Scatter(
        x=sel_df["Date"],
        y=sel_df["Direction"],
        mode="lines+markers",
        name="Direction (1=Up)",
        yaxis="y2",
        line=dict(color="green", dash="dot"),
    )
)
fig_sent.update_layout(
    template="plotly_white",
    height=450,
    xaxis=dict(title="Date", rangeslider=dict(visible=True)),
    yaxis=dict(title="Sentiment"),
    yaxis2=dict(title="Movement", overlaying="y", side="right", range=[-0.2, 1.2]),
)
st.plotly_chart(fig_sent, use_container_width=True)

# Sentiment trend info
if not sel_df.empty:
    trend = "🔺 Bullish" if sel_df["Sentiment"].iloc[-1] > sel_df["Sentiment"].iloc[0] else "🔻 Bearish"
    st.info(f"Overall sentiment trend over selected period: {trend}")

# ────────────────────────────────────────────────────────────────
# Prediction slider & 10‑Day outlook
# ────────────────────────────────────────────────────────────────
st.subheader("🔮 Predict Direction & 10‑Day Outlook")
custom_sent = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.01)
prob = MODEL.predict_proba([[custom_sent]])[0]
label = "Rise" if prob.argmax() == 1 else "Fall"

st.metric(
    label="Prediction",
    value=f"📈 Rise" if label == "Rise" else f"📉 Fall",
    delta=f"{prob.max():.1%} prob.",
)
st.progress(prob.max())

# Future hourly probability (flat forecast demo)
future_hours = pd.date_range(datetime.now(), periods=10 * 24, freq="H")
outlook = pd.DataFrame({"Datetime": future_hours, "ProbRise": prob[1]})
fig_future = go.Figure(
    go.Scatter(x=outlook["Datetime"], y=outlook["ProbRise"], mode="lines", line=dict(color="purple"))
)
fig_future.update_layout(
    template="plotly_white",
    height=350,
    yaxis_title="Probability of Rise",
    xaxis_title="Future (10 days, hourly)",
)
st.plotly_chart(fig_future, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# Download button
# ────────────────────────────────────────────────────────────────
st.download_button(
    "📥 Download Filtered Dataset",
    sel_df.to_csv(index=False),
    file_name=f"{SELECTED_TICKER}_sentiment_{datetime.today().date()}.csv",
)

# Footer
st.caption("Made with ❤️ using Streamlit · Wikipedia · YFinance · VADER · by Debajeet Mandal")
