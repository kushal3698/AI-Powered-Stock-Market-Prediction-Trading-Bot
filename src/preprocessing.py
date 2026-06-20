import os
import argparse
import pandas as pd
import numpy as np

def clean_yfinance_columns(df):
    """
    Flatten multi-level columns if present, and rename columns to standard names.
    """
    # If columns is a multi-index, flatten it
    if isinstance(df.columns, pd.MultiIndex):
        # Often it is (Metric, Ticker) or (Ticker, Metric)
        # Let's check which level has yfinance standard names
        new_cols = []
        for col in df.columns:
            # Join non-empty levels or find standard names
            parts = [c for c in col if c != ""]
            if len(parts) > 1:
                # If one of them is ticker, we keep the metric
                metric = parts[0]
                new_cols.append(metric)
            else:
                new_cols.append(parts[0])
        df.columns = new_cols
    
    # Let's clean names
    df.columns = [str(c).strip() for c in df.columns]
    return df

def add_technical_indicators(df):
    """
    Calculate standard technical indicators.
    """
    # Avoid modifying original dataframe
    df = df.copy()
    
    # Ensure correct sorting by Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    elif df.index.name == 'Date' or (isinstance(df.index, pd.DatetimeIndex)):
        df = df.sort_index()
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Moving Averages
    df['SMA_10'] = close.rolling(window=10).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['EMA_10'] = close.ewm(span=10, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    
    # RSI (14 days)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands (20 days)
    df['BB_Middle'] = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_20 * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_20 * 2)
    
    # ATR (14 days)
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    # Daily Returns
    df['Daily_Return'] = close.pct_change()
    
    # Target: Next day's Daily Return (percentage change)
    df['Target_Return'] = df['Daily_Return'].shift(-1)
    df['Target_Close'] = close.shift(-1)
    
    # Simulate news sentiment correlated with daily return + some random noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(df))
    df['News_Sentiment'] = np.clip(df['Daily_Return'] * 5 + noise, -1.0, 1.0)
    
    return df

def preprocess_pipeline(ticker, input_dir="data/raw", output_dir="data/processed"):
    """
    Load raw data, clean, construct features, and save processed data.
    """
    input_file = os.path.join(input_dir, f"{ticker}.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Raw data file {input_file} not found. Ingest data first.")
        
    print(f"Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Clean columns
    df = clean_yfinance_columns(df)
    
    # Add indicators
    print("Calculating technical indicators...")
    df = add_technical_indicators(df)
    
    # Drop rows with NaN (due to rolling windows)
    # Note: Target_Close will have one NaN at the end, we drop it for training
    # but might want to keep the very last row (without target) for future forecasting.
    # We will save the full dataframe with indicators. For training, we can drop NaNs.
    df_cleaned = df.dropna(subset=['SMA_50', 'ATR_14'])
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{ticker}_processed.csv")
    df_cleaned.to_csv(output_file, index=False)
    
    print(f"Processed data saved to {output_file}")
    print(f"Shape: {df_cleaned.shape}")
    return df_cleaned

def create_sequences(data, seq_length, feature_cols, target_col='Target_Close'):
    """
    Helper function to create sequences for LSTM/Transformer model.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[feature_cols].iloc[i:(i + seq_length)].values)
        y.append(data[target_col].iloc[i + seq_length - 1])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess stock data and generate features")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Directory of raw CSV files")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed CSV files")
    
    args = parser.parse_args()
    
    preprocess_pipeline(args.ticker, args.input_dir, args.output_dir)
