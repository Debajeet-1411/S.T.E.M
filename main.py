import yfinance as yf

# List of ticker symbols
tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "BTC-USD"]

# Download all ticker data at once
df_all = yf.download(
    tickers,
    start="2021-01-01",
    end="2025-06-14",
    interval="3mo",
    group_by="ticker",   # Important for multiple stocks
    auto_adjust=True     # Adjust for splits/dividends
)

# Keep only the 'Close' price for each stock
df_prices = {}

for ticker in tickers:
    df_prices[ticker] = df_all[ticker][["Close"]].rename(columns={"Close": ticker})

# Combine all Close prices into a single DataFrame
from functools import reduce
df_merged = reduce(lambda left, right: left.join(right, how='outer'), df_prices.values())

# Display the final combined table
print(df_merged)
