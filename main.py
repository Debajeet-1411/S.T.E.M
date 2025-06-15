# gdelt_sentiment_pipeline.py
# ────────────────────────────────────────────────────────────────────────────
# Requirements (install once):
#   pip install yfinance pandas requests python-dateutil vaderSentiment
#   pip install scikit-learn joblib tqdm

import os, time, csv, joblib, requests, yfinance as yf, pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─────────────── CONFIG ────────────────────────────────────────────────────
TICKERS     = ["TSLA", "GOOGL", "AAPL", "AMZN", "MSFT"]
CSV_STOCK   = "stock_data.csv"
CSV_NEWS    = "news_head.csv"
MODEL_FILE  = "sentiment_stock_model.joblib"

GDELT_URL   = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_ARTS    = 250          # per‑query cap
REQ_SLEEP   = 0.5          # seconds between API calls (politeness)

# ─────────────── STOCK SECTION ─────────────────────────────────────────────
def update_stock_history():
    """Download any missing OHLC rows & merge into CSV_STOCK."""
    if os.path.exists(CSV_STOCK):
        hist = pd.read_csv(CSV_STOCK, parse_dates=["Date"])
        last_dt = hist["Date"].max().date()
        start_dt = last_dt + timedelta(days=1)
        print(f"[STOCK] Updating from {start_dt} …")
    else:
        hist = pd.DataFrame()
        start_dt = datetime(2022, 1, 1).date()
        print("[STOCK] No CSV found – downloading full history from 2022‑01‑01.")

    end_dt = datetime.today().date()
    if start_dt > end_dt:
        print("[STOCK] File already up‑to‑date.")
        return hist

    df = yf.download(
        TICKERS,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        progress=False
    )
    df.columns = [f"{t}_{c}" for t, c in df.columns]        # flatten MultiIndex
    df = df.reset_index().rename(columns={"Date": "Date"})

    merged = pd.concat([hist, df], ignore_index=True).drop_duplicates(subset=["Date"])
    merged.to_csv(CSV_STOCK, index=False)
    print(f"[STOCK] CSV now holds {len(merged)} trading days.")
    return merged

# ─────────────── NEWS SECTION ──────────────────────────────────────────────
def gdelt_day_query(keyword: str, day: datetime.date):
    """Return list[(ticker, date_str, headline, url)] for that ticker‑day."""
    start = day.strftime("%Y%m%d") + "000000"
    end   = day.strftime("%Y%m%d") + "235959"
    params = dict(query=keyword, mode="ArtList", format="json",
                  maxrecords=MAX_ARTS, startdatetime=start, enddatetime=end)
    resp = requests.get(GDELT_URL, params=params, timeout=30)
    resp.raise_for_status()
    arts = resp.json().get("articles", [])
    return [(keyword, day.strftime("%Y-%m-%d"), a["title"], a.get("url", "")) for a in arts]

def update_news_headlines():
    """
    Ensure news_head.csv has every headline for the last 90 days (per ticker),
    while keeping older ones previously saved.
    """
    today = datetime.today().date()
    start_window = today - timedelta(days=89)

    # Load existing rows into a set for fast de‑duplication
    existing = set()
    if os.path.exists(CSV_NEWS) and os.path.getsize(CSV_NEWS) > 0:
        old = pd.read_csv(CSV_NEWS)
        existing = set(zip(old.Ticker, old.Date, old.Headline))

    new_rows = []
    bar = tqdm(total=90 * len(TICKERS), desc="News fetch")
    for d in (start_window + timedelta(days=i) for i in range(90)):
        for tk in TICKERS:
            try:
                for rec in gdelt_day_query(tk, d):
                    if (rec[0], rec[1], rec[2]) not in existing:
                        new_rows.append(rec)
                time.sleep(REQ_SLEEP)
            except Exception as e:
                print(f"[WARN] {tk} {d}: {e}")
            bar.update(1)
    bar.close()

    # Append new rows
    mode = "a" if os.path.exists(CSV_NEWS) else "w"
    with open(CSV_NEWS, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["Ticker", "Date", "Headline", "URL"])
        w.writerows(new_rows)
    print(f"[NEWS] Added {len(new_rows)} new headlines to {CSV_NEWS}")

    return pd.read_csv(CSV_NEWS, parse_dates=["Date"])

# ─────────────── SENTIMENT MERGE ───────────────────────────────────────────
def build_sentiment_dataset(stock_df: pd.DataFrame, news_df: pd.DataFrame):
    """Return DataFrame [Ticker, Date, Sentiment, Direction]."""
    analyzer = SentimentIntensityAnalyzer()

    # Group all headlines per ticker‑day
    grouped = (news_df.groupby(["Ticker", news_df["Date"].dt.strftime("%Y-%m-%d")])
                       ["Headline"].apply(list)
                       .reset_index())

    rows = []
    for _, rec in grouped.iterrows():
        date_str = rec["Date"]
        if date_str not in stock_df["Date"].astype(str).values:
            continue                                # skip weekends/holidays

        scores = [analyzer.polarity_scores(h)["compound"] for h in rec["Headline"]]
        sentiment = sum(scores) / len(scores)

        open_px  = stock_df.loc[stock_df["Date"] == date_str, f"{rec['Ticker']}_Open"].values[0]
        close_px = stock_df.loc[stock_df["Date"] == date_str, f"{rec['Ticker']}_Close"].values[0]
        direction = 1 if close_px > open_px else 0

        rows.append([rec["Ticker"], date_str, sentiment, direction])

    ds = pd.DataFrame(rows, columns=["Ticker", "Date", "Sentiment", "Direction"])
    print(f"[DATA] Built sentiment dataset: {len(ds)} rows.")
    return ds

# ─────────────── MODEL TRAINING ───────────────────────────────────────────
def train_and_save(df: pd.DataFrame):
    if len(df["Direction"].unique()) < 2:
        print("[MODEL] Need both up & down classes to train.")
        return
    X, y = df[["Sentiment"]].values, df["Direction"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_tr, y_tr)

    print("\n[MODEL] Evaluation")
    print(classification_report(y_te, clf.predict(X_te), digits=3))
    print("Accuracy:", accuracy_score(y_te, clf.predict(X_te)))

    joblib.dump(clf, MODEL_FILE)
    print(f"[MODEL] Saved → {MODEL_FILE}")

# ─────────────── MAIN PIPELINE ────────────────────────────────────────────
if __name__ == "__main__":
    stock_df = update_stock_history()
    news_df  = update_news_headlines()
    ds       = build_sentiment_dataset(stock_df, news_df)
    if ds.empty:
        print("[PIPE] No overlapping sentiment‑price rows yet.")
    else:
        train_and_save(ds)
