# AI Stock‑&‑Crypto Price Predictor

> **Predict next‑quarter prices for multiple assets (Apple, Microsoft, Tesla, Alphabet & Bitcoin) using 3‑month historical data.**

---

## 🚀 Project Overview

This repo demonstrates a **minimal, end‑to‑end pipeline** that:

1. **Downloads** 3‑month OHLCV data from Yahoo Finance (`yfinance`).
2. **Pre‑processes** it into sliding windows (last 4 quarters → next quarter).
3. **Trains** a baseline model (Random Forest Regressor).
4. **Predicts** the next 3‑month closing price for each ticker.

The goal is to serve as a *learning scaffold*—you can swap in news‑sentiment features, LSTMs, Transformers, AutoML, etc.

---

## ✨ Key Features

| Feature              | Details                                   |
| -------------------- | ----------------------------------------- |
| Multi‑asset download | AAPL, MSFT, TSLA, GOOGL, BTC‑USD          |
| Custom date window   | 2021‑01‑01 → 2025‑06‑14 by default        |
| 3‑month interval     | `interval="3mo"` for macro‑trend focus    |
| Baseline ML model    | `sklearn.RandomForestRegressor`           |
| Sliding‑window prep  | Past 4 quarters → next quarter target     |
| Reproducible env     | `requirements.txt` + `.venv` instructions |

---

## 🗂️ Project Structure

```text
📂 ai‑price‑predictor/
├─ README.md            ←‑ you are here
├─ requirements.txt     ←‑ packages (numpy, pandas, yfinance, scikit‑learn)
├─ data/                ←‑ cached price CSVs (auto‑created)
├─ notebooks/
│   └─ 01_quickstart.ipynb
├─ src/
│   ├─ fetch.py         ←‑ download & cache raw data
│   ├─ features.py      ←‑ build sliding windows
│   ├─ model.py         ←‑ train + predict + save model
│   └─ main.py          ←‑ CLI glue (fetch → train → predict)
└─ outputs/
    └─ predictions.csv  ←‑ next‑quarter forecast
```

---

## 🛠️ Requirements

```bash
Python >= 3.9
pip >= 23
```

All dependencies are open‑source and free.

---

## ⚙️ Installation

```bash
# 1. clone
git clone https://github.com/yourname/ai-price-predictor.git
cd ai-price-predictor

# 2. create + activate virtual env (Windows example)
python -m venv .venv
.venv\Scripts\Activate.ps1  # or .venv\Scripts\activate.bat

# 3. install deps
pip install --upgrade pip
pip install -r requirements.txt
```

*If PowerShell blocks script activation, run* `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` *once.*

---

## 🏃 Quick Start (CLI)

```bash
python src/main.py --tickers AAPL MSFT TSLA GOOGL BTC-USD \
                   --start 2021-01-01 --end 2025-06-14 \
                   --interval 3mo

cat outputs/predictions.csv
```

---

## 📑 How It Works

1. `` pulls raw OHLCV data via `yfinance.download()` with `group_by="ticker"` and `auto_adjust=True`.
2. `` constructs supervised pairs `(X, y)` where:
   - `X` = prices for quarters *t‑4 … t‑1*
   - `y` = price at quarter *t*
3. `` fits a `RandomForestRegressor` and saves a `joblib` model.
4. `` orchestrates the pipeline and writes `outputs/predictions.csv`.

---

## 📈 Sample Prediction (Apple)

```text
Date,Actual,Predicted
2024-07-01,195.56,193.12
2024-10-01,189.89,191.77
🔮 Next (2025‑07‑01),–,198.34
```

> *Baseline RMSE ≈ 5.8. Expect better accuracy once you add sentiment or tech indicators.*

---

## 🛣️ Next Steps & Ideas

- **Add news sentiment** (HuggingFace FinBERT) to features.
- Swap model for **LSTM** or **Temporal Fusion Transformer**.
- Tune hyper‑params with **Optuna**.
- Deploy as **FastAPI** microservice + web dashboard.

---

## 🤝 Contributing

Pull requests welcome! Please open an issue first to discuss what you’d like to change.

---

## 🪪 License

MIT — see `LICENSE` file for details.

