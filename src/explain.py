import os
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

FEATURE_COLS = [
    'Close', 'Volume', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'ATR_14', 'Daily_Return'
]

def generate_shap_plots(ticker, processed_file="data/processed/{ticker}_processed.csv", models_dir="models"):
    """
    Calculate SHAP values for the baseline Random Forest model and save the summary plot as an image.
    """
    filepath = processed_file.format(ticker=ticker)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed file {filepath} not found.")
        
    df = pd.read_csv(filepath)
    
    # Load model and scaler
    rf_path = os.path.join(models_dir, f"{ticker}_rf.pkl")
    scaler_path = os.path.join(models_dir, f"{ticker}_scaler.pkl")
    
    if not os.path.exists(rf_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or Scaler file not found. Train the model first.")
        
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Scale dataset
    all_cols = FEATURE_COLS + ['Target_Close']
    df_clean = df.dropna(subset=all_cols).reset_index(drop=True)
    
    scaled_features = scaler.transform(df_clean[FEATURE_COLS])
    X = pd.DataFrame(scaled_features, columns=FEATURE_COLS)
    
    # Initialize TreeExplainer
    print("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(rf_model)
    
    # Sample a subset to speed up computation if needed (e.g. latest 200 rows)
    X_sample = X.tail(200)
    shap_values = explainer.shap_values(X_sample)
    
    # Create SHAP summary plot
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    # Turn off interactive plotting
    plt.ioff()
    
    # Generate the plot
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_COLS, show=False)
    
    # Save the plot
    output_path = os.path.join(models_dir, f"{ticker}_shap_summary.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP summary plot successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for models")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--processed_file", type=str, default="data/processed/{ticker}_processed.csv", help="Template path to processed CSV file")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save models and plots")
    
    args = parser.parse_args()
    
    generate_shap_plots(args.ticker, args.processed_file, args.models_dir)
