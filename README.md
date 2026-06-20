# AI-Powered Stock Market Prediction & Trading Bot

A modular Python starter project for building an AI/ML-powered stock prediction and trading assistant. It automates stock data ingestion, technical indicators feature engineering, news sentiment parsing, deep learning forecasting, strategy backtesting, and model explainability, wrapped in a professional Streamlit dashboard.

## Features

- **Stock Data Ingestion**: Seamless ingestion of historical ticker data from Yahoo Finance (`yfinance`).
- **Feature Engineering**: Built-in calculations for standard technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, Average True Range) using Pandas/Numpy (no compiled C-libraries needed).
- **Sentiment Analysis**: Dynamic financial news scraping and sentiment polarity scoring using NLTK's VADER model.
- **AI Forecasting Models**:
  - *Baseline*: Scikit-Learn Random Forest Regressor trained on return percentage targets.
  - *Neural Network (MLP)*: Sequence-aware Multi-Layer Perceptron (`MLPRegressor`) using target scaling (x100) for high-accuracy, zero-dependency convergence.
  - *LSTM (Deep Learning)*: Sequential model in Keras/TensorFlow.
  - *Transformer (Advanced)*: Multi-head attention-based encoder architecture for sequence modeling.
- **Model Voting System (Consensus Trading)**: Multi-model strategy that only triggers trade actions (BUY/SELL) if both the Random Forest and MLP models agree on tomorrow's direction (helps filter noise and overfitting).
- **SHAP Feature Selection**: Pre-training pipeline automatically prunes features with low mean absolute SHAP values (<0.001) to keep models clean and noise-free.
- **Trading Backtester**: Simulates portfolio actions (BUY, SELL, HOLD) based on predictions and evaluates performance metrics (Sharpe Ratio, Max Drawdown, Win Rate, cumulative returns).
- **Explainability**: Computes SHAP value attributions on baseline model features to offer explainable forecasts.
- **Streamlit Web Dashboard**: Interactive, modern dark-themed dashboard to visualize market trends, AI predictions, sentiment metrics, backtest results, and SHAP summaries.

---

## Project Structure

```text
.
├── data/
│   ├── raw/                 # Downloaded raw CSV data from yFinance
│   └── processed/           # Processed datasets with technical indicators
├── models/                  # Saved weights, scalers, and explanation plots
├── notebooks/               # For experimentation
├── src/
│   ├── __init__.py
│   ├── data_ingest.py       # Data downloading script
│   ├── preprocessing.py     # Indicator math and sequences processing
│   ├── sentiment.py         # Financial news scraping and NLTK sentiment analysis
│   ├── models.py            # Random Forest, LSTM, and Transformer model classes
│   ├── backtest.py          # Portfolio simulation and metric generation
│   ├── explain.py           # SHAP value calculations and summaries
│   └── dashboard.py         # Streamlit UI dashboard
├── Makefile                 # Easy pipeline shortcuts
└── requirements.txt         # Project requirements
```

---

## Installation & Setup

1. **Clone/Setup Folder**:
   Initialize a Python 3.13 virtual environment:
   ```bash
   python -m venv .venv
   # On Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # On macOS/Linux
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Use (Command Line)

### 1. Ingest Data
Download historical price data for a ticker:
```bash
python src/data_ingest.py --ticker AAPL --start 2020-01-01 --end 2024-01-01
```

### 2. Preprocess & Add Indicators
Create engineered indicators:
```bash
python src/preprocessing.py --ticker AAPL
```

### 3. Train Models
Train Random Forest, LSTM, and Transformer models:
```bash
python src/models.py --ticker AAPL
```

### 4. Generate SHAP Explanations
Create SHAP summary plots:
```bash
python src/explain.py --ticker AAPL
```

### 5. Launch the Dashboard
Run the Streamlit dashboard:
```bash
streamlit run src/dashboard.py
```

---

## Makefile Shortcuts

Use the following `make` commands:
- `make install` - Installs all requirements.
- `make ingest TICKER=AAPL START=2020-01-01 END=2024-01-01` - Ingests historical data.
