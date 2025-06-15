import os
import requests
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ────── CONFIGURATION ─────────────────────────────────────────
TICKERS = ["TSLA", "GOOGL", "AAPL", "AMZN", "MSFT"]
CSV_STOCK = "stock_data.csv"
CSV_NEWS = "NEWS_HEAD.csv"
CSV_FINAL = "sentiment_vs_stock.csv"
MODEL_FILE = "sentiment_stock_model.joblib"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # Set this env variable or hardcode your key
NEWDATA_API_KEY = "YOUR_NEWSDATA_API_KEY"  # Replace with your NewsData.io API key

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

# ────── 2. FETCH HEADLINES ───────────────────────────────────
def fetch_headlines_for_date(date: str, ticker: str):
    date_str = date.replace("-", "")
    search_term = f'"{ticker}"'
    headlines = []

    # ───── GDELT TRY ─────
    gdelt_url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc?query={search_term}"
        f"&mode=artlist&maxrecords=100&format=json&startdatetime={date_str}000000"
        f"&enddatetime={date_str}235959"
    )

    try:
        response = requests.get(gdelt_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "articles" in data:
                headlines = [a["title"] for a in data["articles"] if "title" in a]
                if headlines:
                    return headlines
        else:
            print(f"⚠️  GDELT returned status {response.status_code} for {ticker} on {date}")
    except Exception as e:
        print(f"⚠️  GDELT failed for {ticker} on {date}: {e}")

    # ───── NewsAPI TRY ─────
    if NEWSAPI_KEY:
        try:
            url = (
                f"https://newsapi.org/v2/everything?q={ticker}&from={date}&to={date}"
                f"&sortBy=publishedAt&language=en&apiKey={NEWSAPI_KEY}"
            )
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                headlines = [a["title"] for a in articles if "title" in a]
                if headlines:
                    return headlines
            else:
                print(f"❌ NewsAPI error {response.status_code} for {ticker} on {date}")
        except Exception as e:
            print(f"❌ NewsAPI fetch failed for {ticker} on {date}: {e}")
    else:
        print(f"❌ No NewsAPI key set.")

    # ───── NewsData.io FALLBACK ─────
    try:
        url = (
            f"https://newsdata.io/api/1/news?apikey={NEWDATA_API_KEY}"
            f"&q={ticker}&language=en&from_date={date}&to_date={date}&category=business"
        )
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            headlines = [item["title"] for item in results if "title" in item]
            if headlines:
                print(f"✅ NewsData.io fallback used for {ticker} on {date}")
                return headlines
            else:
                print(f"⚠️ NewsData.io found no articles for {ticker} on {date}")
        else:
            print(f"❌ NewsData.io error {response.status_code} for {ticker} on {date}")
    except Exception as e:
        print(f"❌ NewsData.io fetch failed for {ticker} on {date}: {e}")

    return []

# ────── 3. UPDATE NEWS CACHE ─────────────────────────────────
def update_news_cache(stock_df):
    if os.path.exists(CSV_NEWS):
        news_df = pd.read_csv(CSV_NEWS)
    else:
        news_df = pd.DataFrame(columns=["Ticker", "Date", "Headline"])

    new_rows = []
    all_dates = stock_df["Date"].astype(str).unique()

    for ticker in TICKERS:
        existing = news_df[(news_df.Ticker == ticker)]["Date"].unique()
        missing = sorted(set(all_dates) - set(existing))

        for date in missing:
            headlines = fetch_headlines_for_date(date, ticker)
            for h in headlines:
                new_rows.append({"Ticker": ticker, "Date": date, "Headline": h})

    if new_rows:
        news_df = pd.concat([news_df, pd.DataFrame(new_rows)], ignore_index=True)
        news_df.to_csv(CSV_NEWS, index=False)
        print(f"📰 Added {len(new_rows)} new headlines to {CSV_NEWS}.")
    else:
        print("✅ No new headlines to fetch.")

    return news_df

# ────── 4. BUILD SENTIMENT DATASET ───────────────────────────
def build_sentiment_dataset(stock_df, news_df):
    analyzer = SentimentIntensityAnalyzer()
    rows = []

    grouped = news_df.groupby(["Ticker", "Date"])

    for (ticker, date), group in grouped:
        headlines = group["Headline"].tolist()
        sentiments = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        try:
            open_px = stock_df.loc[stock_df["Date"] == date, f"{ticker}_Open"].values[0]
            close_px = stock_df.loc[stock_df["Date"] == date, f"{ticker}_Close"].values[0]
            direction = 1 if close_px > open_px else 0
            rows.append([ticker, date, avg_sentiment, direction])
        except IndexError:
            continue

    result_df = pd.DataFrame(rows, columns=["Ticker", "Date", "Sentiment", "Direction"])
    result_df.to_csv(CSV_FINAL, index=False)
    print(f"💾 Saved sentiment data to {CSV_FINAL} with {len(result_df)} rows.")
    return result_df

# ────── 5. TRAIN & SAVE MODEL ────────────────────────────────
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
    print("🚀 Starting Sentiment-Stock Pipeline")
    stock_df = update_stock_history()
    news_df = update_news_cache(stock_df)
    sentiment_df = build_sentiment_dataset(stock_df, news_df)
    train_model(sentiment_df)
    print("✅ Pipeline finished.")
