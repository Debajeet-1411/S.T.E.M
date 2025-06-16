import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import joblib

CSV_STOCK = "store_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TICKER_MAP = {
    "AAPL": "Apple Inc.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "MSFT": "Microsoft Corporation",
    "TSLA": "Tesla Inc."
}


def update_stock_history():
    tickers = list(TICKER_MAP.keys())
    start = "2022-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    all_data = []
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end)
        data = data.reset_index()
        data["Ticker"] = ticker
        all_data.append(data)

    stock_df = pd.concat(all_data)
    stock_df.to_csv(CSV_STOCK, index=False)
    return stock_df


def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity


def build_sentiment_dataset(stock_df):
    # Simulate dummy sentiment scores and headlines
    stock_df = stock_df.copy()
    stock_df["Headline"] = stock_df["Date"].astype(str) + " news for " + stock_df["Ticker"]
    stock_df["Sentiment"] = stock_df["Headline"].apply(calculate_sentiment)

    # Add direction
    stock_df.sort_values(["Ticker", "Date"], inplace=True)
    stock_df["Next Close"] = stock_df.groupby("Ticker")["Close"].shift(-1)
    stock_df["Direction"] = (stock_df["Next Close"] > stock_df["Close"]).astype(int)

    # Drop rows with NaN
    stock_df.dropna(subset=["Next Close"], inplace=True)

    return stock_df[["Date", "Ticker", "Close", "Volume", "High", "Low", "Sentiment", "Direction"]]


def add_features(df):
    df = df.copy()
    df["Volatility"] = df["High"] - df["Low"]
    df["Price Change"] = df["Close"].diff()
    df["Rolling Sentiment"] = df.groupby("Ticker")["Sentiment"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df.dropna(inplace=True)
    return df


def train_models(df):
    models = {}
    for ticker in df["Ticker"].unique():
        df_ticker = df[df["Ticker"] == ticker].copy()

        # Balance classes
        df_majority = df_ticker[df_ticker["Direction"] == 1]
        df_minority = df_ticker[df_ticker["Direction"] == 0]
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
        df_balanced = pd.concat([df_majority, df_minority_upsampled])

        # Train model
        X = df_balanced[["Sentiment", "Volatility", "Price Change", "Rolling Sentiment"]]
        y = df_balanced["Direction"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ {ticker} Model Accuracy: {acc:.2f}")
        print(classification_report(y_test, y_pred))

        joblib.dump(model, f"{MODEL_DIR}/{ticker}_model.joblib")
        models[ticker] = model
    return models


if __name__ == "__main__":
    print("🔄 Updating stock history...")
    stock_df = update_stock_history()

    print("🧠 Building dataset with sentiment and features...")
    sentiment_df = build_sentiment_dataset(stock_df)
    feature_df = add_features(sentiment_df)

    print("🚀 Training per-stock models...")
    train_models(feature_df)
    print("✅ All models trained and saved in 'models/' directory.")
