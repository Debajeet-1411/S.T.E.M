import os
import pandas as pd
import joblib
import yfinance as yf
import wikipedia
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ────── CONFIGURATION ─────────────────────────────────────────
TICKERS = ["TSLA", "GOOGL", "AAPL", "AMZN", "MSFT"]
CSV_STOCK = "stock_data.csv"
CSV_WIKI = "wiki_sentiment.csv"
MODEL_FILE = "sentiment_stock_model.joblib"

# ────── 1. UPDATE STOCK HISTORY ──────────────────────────────
def update_stock_history():
    if os.path.exists(CSV_STOCK):
        hist = pd.read_csv(CSV_STOCK, parse_dates=["Date"])
        last_date = hist["Date"].max().date()
        start_dt = last_date + timedelta(days=1)
    else:
        hist = pd.DataFrame()
        start_dt = datetime(2022, 1, 1).date()

    end_dt = datetime.today().date()
    if start_dt > end_dt:
        print("✅ Stock data already up to date.")
        return hist

    df = yf.download(
        TICKERS,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        progress=False
    )

    df.columns = [f"{ticker}_{col}" for ticker, col in df.columns]
    df = df.reset_index().rename(columns={"Date": "Date"})
    merged = pd.concat([hist, df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"])
    merged.to_csv(CSV_STOCK, index=False)
    print(f"📈 Stock data updated with {len(merged) - len(hist)} new rows.")
    return merged

# ────── 2. FETCH WIKIPEDIA CONTENT ───────────────────────────
def fetch_wikipedia_summary(ticker):
    name_map = {
        "TSLA": "Tesla, Inc.",
        "GOOGL": "Alphabet Inc.",
        "AAPL": "Apple Inc.",
        "AMZN": "Amazon (company)",
        "MSFT": "Microsoft"
    }
    try:
        title = name_map.get(ticker, ticker)
        summary = wikipedia.page(title).content
        return summary[:3000]  # Limit to 3000 characters
    except Exception as e:
        print(f"❌ Wikipedia fetch failed for {ticker}: {e}")
        return ""

# ────── 3. BUILD SENTIMENT DATASET ───────────────────────────
def build_sentiment_dataset(stock_df):
    analyzer = SentimentIntensityAnalyzer()
    rows = []

    for ticker in TICKERS:
        text = fetch_wikipedia_summary(ticker)
        sentiment = analyzer.polarity_scores(text)["compound"] if text else 0

        try:
            df = stock_df[["Date", f"{ticker}_Open", f"{ticker}_Close"]].dropna()
            for _, row in df.iterrows():
                direction = 1 if row[f"{ticker}_Close"] > row[f"{ticker}_Open"] else 0
                rows.append([ticker, row["Date"], sentiment, direction])
        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

    result_df = pd.DataFrame(rows, columns=["Ticker", "Date", "Sentiment", "Direction"])
    result_df.to_csv(CSV_WIKI, index=False)
    print(f"💾 Saved Wikipedia sentiment data to {CSV_WIKI} with {len(result_df)} rows.")
    return result_df

# ────── 4. TRAIN & SAVE MODEL ────────────────────────────────
def train_model(df):
    if df.empty or len(df["Direction"].unique()) < 2:
        print("⚠️  Not enough data to train model.")
        return

    X = df[["Sentiment"]].values
    y = df["Direction"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(clf, MODEL_FILE)
    print(f"✅ Model saved to {MODEL_FILE}")

# ────── MAIN EXECUTION ───────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting Wikipedia Sentiment-Stock Pipeline")
    stock_df = update_stock_history()
    sentiment_df = build_sentiment_dataset(stock_df)
    train_model(sentiment_df)
    print("✅ Pipeline finished.")
