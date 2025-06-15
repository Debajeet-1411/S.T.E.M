
# 📊 Stock Sentiment Analyzer using Wikipedia (GSoC 2025 Project)

This project is a pipeline that fetches historical stock data, extracts sentiment from Wikipedia content related to selected companies, and trains a logistic regression model to predict stock movement direction based on sentiment. It also includes a user-friendly **Streamlit interface** for interaction and visualization.

---

## 🚀 Features

- ✅ Fetches and updates historical stock data daily using Yahoo Finance
- ✅ Extracts Wikipedia summaries for major tech companies
- ✅ Performs sentiment analysis using VADER on the Wikipedia content
- ✅ Builds a dataset linking sentiment scores to stock price movements
- ✅ Trains a Logistic Regression model to predict upward or downward price movement
- ✅ Saves trained model using `joblib`
- ✅ Provides a responsive **web UI** via Streamlit
- ✅ Automatically runs the update pipeline every day at **6 PM**
- 🔄 Future scope: pattern-based predictions from stock graphs

---

## 📂 Folder Structure

```
GSOC/
├── main.py                 # Main pipeline script (data + training)
├── app.py                  # Streamlit app for UI
├── sentiment_stock_model.joblib  # Trained ML model
├── stock_data.csv          # Historical stock data
├── wiki_sentiment.csv      # Sentiment + direction dataset
├── schedule_task.bat       # Scheduler script for Windows (optional)
├── README.md               # You are here!
```

---

## 🛠️ Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:
```text
pandas
yfinance
wikipedia
scikit-learn
vaderSentiment
joblib
streamlit
```

---

## 💡 How It Works

### 1. Update Stock History
Downloads daily stock data for the tickers:
```python
["TSLA", "GOOGL", "AAPL", "AMZN", "MSFT"]
```
Using Yahoo Finance (`yfinance`) and stores in `stock_data.csv`.

---

### 2. Fetch Wikipedia Sentiment
For each company, fetches up to 3000 characters from its Wikipedia summary and runs VADER sentiment analysis. Example:

- **Ticker**: `TSLA`
- **Sentiment Score**: `0.63`

---

### 3. Train Model
Builds a dataset with:
- Date
- Ticker
- Sentiment Score
- Direction (1 if Close > Open, else 0)

Then trains a **Logistic Regression model** using `scikit-learn`.

---

### 4. Run the Web App

```bash
streamlit run app.py
```

Features:
- Button to run the full pipeline (update → sentiment → train)
- Shows success messages and logs
- Ready to integrate graphs, predictions, and more!

---

## 🔁 Automating Daily Updates

A scheduled task runs `main.py` every day at 6 PM.

You can use:
- `schedule` module for Python automation
- `cron` (Linux/macOS)
- `Task Scheduler` (Windows with `.bat` file)

---

## 📌 Future Plans

- 📈 Graph-based pattern recognition and predictions
- 📉 Compare sentiment predictions with actual market movement
- 📊 Visual analytics dashboard in Streamlit
- 🌐 Optional API support for integrating external sources (e.g., GDELT, NewsAPI)

---

## 👨‍💻 Author

**Debajeet Mandal**  
*Google Summer of Code 2025 Contributor*

---

## 📃 License

This project is open-sourced under the MIT License.
