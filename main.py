import yfinance as yf

ticker = "AAPL"

# Download data from Jan 1, 2021 to today with 3-month interval
df_price = yf.download(
    ticker,
    start="2021-01-01",
    end="2025-06-14",           # Or use end="today" for dynamic latest
    interval="3mo"              # 3-month interval
)

# Optional: Keep only the Close column and rename it to 'price'
df_price = df_price[["Close"]].rename(columns={"Close": "price"})

# Show result
print(df_price)
