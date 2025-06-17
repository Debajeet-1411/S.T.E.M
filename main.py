#!/usr/bin/env python3
"""
Wikipedia‑Sentiment → Next‑day Stock‑Direction pipeline
------------------------------------------------------

Steps
1. Incrementally update daily OHLC data for the configured tickers.
2. Fetch (and cache) short Wikipedia summaries & compute VADER sentiment.
3. Build a dataset whose label is *next‑day* price direction (eliminates look‑ahead bias).
4. Chronological 80 / 20 split → LogisticRegression.
Outputs
    data/stock_data.csv
    data/wiki_sentiment.csv
    models/sentiment_stock_model.joblib
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
import wikipedia
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ───────────────────────────────
# CONFIG
# ───────────────────────────────
TICKERS: List[str] = ["TSLA", "GOOGL", "AAPL", "AMZN", "MSFT"]

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CSV_STOCK = DATA_DIR / "stock_data.csv"
CSV_WIKI = DATA_DIR / "wiki_sentiment.csv"
MODEL_FILE = MODEL_DIR / "sentiment_stock_model.joblib"
WIKI_CACHE_FILE = DATA_DIR / "wiki_cache.json"

START_DATE = date(2022, 1, 1)         # first back‑fill date
SUMMARY_SENTENCES = 10                 # sentences pulled from Wikipedia
TEST_SPLIT_FRAC = 0.20                 # 80 % train | 20 % test


# ───────────────────────────────
# 1 ▸ UPDATE STOCK HISTORY
# ───────────────────────────────
def update_stock_history() -> pd.DataFrame:
    """Download any missing daily OHLC rows and merge into `stock_data.csv`."""
    if CSV_STOCK.exists():
        hist = pd.read_csv(CSV_STOCK, parse_dates=["Date"])
        last_dt: date = hist["Date"].max().date()
        start_dt = (last_dt + pd.tseries.offsets.BDay()).date()  # next market day
    else:
        hist = pd.DataFrame()
        start_dt = START_DATE

    end_dt = datetime.today().date()
    if start_dt > end_dt:
        print("✅ Stock data already up‑to‑date.")
        return hist

    print(f"⬇️  Fetching prices {start_dt} → {end_dt} …")
    df = yf.download(
        tickers=TICKERS,
        start=start_dt,
        end=(end_dt + pd.Timedelta(days=1)),
        interval="1d",
        progress=False,
    )

    if df.empty:
        print("⚠️  No rows returned (market holiday or network issue).")
        return hist

    # Flatten MultiIndex columns: (attribute, ticker) → ticker_attribute
    df.columns = [f"{t}_{f}" for f, t in df.columns]   # (‘Open’, ‘TSLA’) → TSLA_Open
    df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    merged = (
        pd.concat([hist, df], ignore_index=True)
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    merged.to_csv(CSV_STOCK, index=False)
    print(f"✅ Stock CSV now holds {len(merged)} rows.")
    return merged


# ───────────────────────────────
# 2 ▸ WIKIPEDIA SENTIMENT
# ───────────────────────────────
def _load_wiki_cache() -> Dict[str, str]:
    if WIKI_CACHE_FILE.exists():
        try:
            return json.loads(WIKI_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_wiki_cache(cache: Dict[str, str]) -> None:
    WIKI_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def fetch_wikipedia_summary(ticker: str, cache: Dict[str, str]) -> str:
    """Resolve ticker → page title, fetch summary, cache it."""
    if ticker in cache:
        return cache[ticker]

    title_map = {
        "TSLA": "Tesla, Inc.",
        "GOOGL": "Alphabet Inc.",
        "AAPL": "Apple Inc.",
        "AMZN": "Amazon (company)",
        "MSFT": "Microsoft",
    }
    title = title_map.get(ticker, ticker)
    try:
        text = wikipedia.summary(title, sentences=SUMMARY_SENTENCES, auto_suggest=False)
        cache[ticker] = text
        _save_wiki_cache(cache)
        return text
    except Exception as e:
        print(f"❌ Wikipedia fetch failed for {ticker} ({title}): {e}")
        cache[ticker] = ""
        _save_wiki_cache(cache)
        return ""


def build_sentiment_dataset(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker:
      • Compute sentiment of its (cached) Wikipedia page.
      • For every trading day, label whether *next‑day* close > today’s close.
    Saves the result → `wiki_sentiment.csv`.
    """
    analyzer = SentimentIntensityAnalyzer()
    cache = _load_wiki_cache()
    records: List[Tuple] = []

    for ticker in TICKERS:
        wiki_text = fetch_wikipedia_summary(ticker, cache)
        score = analyzer.polarity_scores(wiki_text)["compound"] if wiki_text else 0.0

        col_close = f"{ticker}_Close"
        if col_close not in stock_df.columns:
            print(f"⚠️  Missing Close column for {ticker} — skipped.")
            continue

        df = stock_df[["Date", col_close]].dropna().copy()
        df["Next_Close"] = df[col_close].shift(-1)
        df = df.iloc[:-1]  # last row has no next‑day close
        df["Direction"] = (df["Next_Close"] > df[col_close]).astype(int)

        for _, r in df.iterrows():
            records.append((ticker, r["Date"].date(), score, int(r["Direction"])))

    sentiment_df = pd.DataFrame(
        records, columns=["Ticker", "Date", "Sentiment", "Direction"]
    )
    sentiment_df.to_csv(CSV_WIKI, index=False)
    print(f"✅ Sentiment dataset written ({len(sentiment_df)} rows).")
    return sentiment_df


# ───────────────────────────────
# 3 ▸ TRAIN MODEL
# ───────────────────────────────
def train_model(df: pd.DataFrame) -> None:
    if df.empty or len(df["Direction"].unique()) < 2:
        print("⚠️  Not enough labelled rows to train.")
        return

    df = df.sort_values("Date")
    split_idx = int(len(df) * (1 - TEST_SPLIT_FRAC))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[["Sentiment"]].values, train_df["Direction"].values
    X_test, y_test = test_df[["Sentiment"]].values, test_df["Direction"].values

    clf = LogisticRegression(max_iter=1_000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n📊 Hold‑out evaluation:")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    joblib.dump(clf, MODEL_FILE)
    print(f"✅ Model saved → {MODEL_FILE}")


# ───────────────────────────────
# 4 ▸ MAIN
# ───────────────────────────────
def main() -> None:
    print("\n🚀 Wikipedia‑Sentiment → Stock‑Direction pipeline")
    stock_df = update_stock_history()
    sentiment_df = build_sentiment_dataset(stock_df)
    train_model(sentiment_df)
    print("🏁 Done.")


if __name__ == "__main__":
    main()
