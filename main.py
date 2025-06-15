import yfinance as yf
from datetime import datetime
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Get today's date dynamically
today = datetime.today().strftime('%Y-%m-%d')
# Get historical data for Tesla, Google, Apple, Amazon and Microsoft
data = yf.download(["TSLA", "GOOGL", "AAPL", "AMZN", "MSFT"], start="2022-01-01", end=today, interval="1d")
data.to_csv("stock_data.csv")
print(data.head())
print("Stock data saved to 'stock_data.csv'")





