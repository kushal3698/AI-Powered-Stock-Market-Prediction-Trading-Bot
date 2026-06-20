import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None

# Set random seeds for reproducibility
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)

def train_and_prune(X, y):
    """
    Train base Random Forest model and prune features based on mean absolute SHAP values.
    Keeps features with impact > 0.001.
    """
    import shap
    print("Training base Random Forest model for feature selection...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Use SHAP to calculate importance on a subset of samples for performance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:200])
    
    # Calculate mean absolute SHAP value for each feature (handles numpy arrays and Explanation objects)
    if hasattr(shap_values, "values"):
        vals = np.abs(shap_values.values).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)
        
    kept_features = [col for col, val in zip(X.columns, vals) if val > 0.001]
    print(f"Features kept: {kept_features}")
    return X[kept_features], kept_features, model

FEATURE_COLS = [
    'Close', 'EMA_10', 'EMA_50', 
    'RSI_14', 'MACD', 'MACD_Signal', 
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'ATR_14', 'Daily_Return',
    'News_Sentiment'
]

def prepare_data(df, seq_length=10, test_split=0.2, feature_cols=None):
    """
    Prepare datasets for baseline and deep learning models using percentage returns as the target.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
        
    # Drop rows with NaNs in feature/target columns
    all_cols = feature_cols + ['Target_Return']
    df_clean = df.dropna(subset=all_cols).reset_index(drop=True)
    
    # Time-series split (no random shuffling to avoid data leakage)
    split_idx = int(len(df_clean) * (1 - test_split))
    
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    # Fit scaler on training features only
    scaler = StandardScaler()
    train_scaled_features = scaler.fit_transform(train_df[feature_cols])
    test_scaled_features = scaler.transform(test_df[feature_cols])
    
    # Reconstruct scaled dataframes for sequence extraction
    train_scaled_df = pd.DataFrame(train_scaled_features, columns=feature_cols)
    train_scaled_df['Target_Return'] = train_df['Target_Return'].values
    
    test_scaled_df = pd.DataFrame(test_scaled_features, columns=feature_cols)
    test_scaled_df['Target_Return'] = test_df['Target_Return'].values
    
    # Create sequential datasets for LSTM/Transformer
    from preprocessing import create_sequences
    X_train_seq, y_train_seq = create_sequences(train_scaled_df, seq_length, feature_cols, 'Target_Return')
    X_test_seq, y_test_seq = create_sequences(test_scaled_df, seq_length, feature_cols, 'Target_Return')
    
    # Baseline models use the raw/flat features (last element of each sequence for 1-step, or simply standard tabular dataset)
    X_train_flat = X_train_seq[:, -1, :]
    y_train_flat = y_train_seq
    X_test_flat = X_test_seq[:, -1, :]
    y_test_flat = y_test_seq
    
    # Align dates and close prices correctly
    test_raw_dates = df_clean['Date'].iloc[split_idx + seq_length - 1 : split_idx + seq_length - 1 + len(y_test_seq)].values
    test_raw_close = df_clean['Close'].iloc[split_idx + seq_length - 1 : split_idx + seq_length - 1 + len(y_test_seq)].values
    test_raw_target_close = df_clean['Target_Close'].iloc[split_idx + seq_length - 1 : split_idx + seq_length - 1 + len(y_test_seq)].values
    
    return {
        "X_train_flat": X_train_flat,
        "y_train_flat": y_train_flat,
        "X_test_flat": X_test_flat,
        "y_test_flat": y_test_flat,
        "X_train_seq": X_train_seq,
        "y_train_seq": y_train_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
        "scaler": scaler,
        "test_raw_dates": test_raw_dates,
        "test_raw_close": test_raw_close,
        "test_raw_target_close": test_raw_target_close
    }

def build_lstm_model(input_shape):
    """
    Build Keras LSTM forecasting model.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build LSTM model.")
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

def build_transformer_model(input_shape, head_size=64, num_heads=2, ff_dim=64, num_transformer_blocks=2, mlp_units=[32], dropout=0.1, mlp_dropout=0.1):
    """
    Build Keras Time-Series Transformer forecasting model.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build Transformer model.")
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x_norm, x_norm)
        x_attn = layers.Dropout(dropout)(attn_output)
        x = layers.Add()([x_attn, x])

        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        ff_output = layers.Dense(ff_dim, activation="relu")(x_norm)
        ff_output = layers.Dropout(dropout)(ff_output)
        ff_output = layers.Dense(input_shape[-1])(ff_output)
        x = layers.Add()([ff_output, x])

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

def train_pipeline(ticker, input_file="data/processed/{ticker}_processed.csv", models_dir="models"):
    """
    Train baseline RF, MLP Neural Network, LSTM, and Transformer models, and save them.
    Includes a dynamic SHAP feature selection step.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    filepath = input_file.format(ticker=ticker)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed file {filepath} not found.")
        
    df = pd.read_csv(filepath)
    
    # Step A: Run base Random Forest and prune features using SHAP
    data_init = prepare_data(df, feature_cols=FEATURE_COLS)
    X_train_df = pd.DataFrame(data_init["X_train_flat"], columns=FEATURE_COLS)
    y_train_series = pd.Series(data_init["y_train_flat"])
    
    X_pruned, kept_features, rf_init = train_and_prune(X_train_df, y_train_series)
    
    # Save kept features list
    kept_features_path = os.path.join(models_dir, f"{ticker}_kept_features.pkl")
    with open(kept_features_path, 'wb') as f:
        pickle.dump(kept_features, f)
    print(f"Kept features saved to {kept_features_path}")
    
    # Step B: Prepare final datasets using only kept features
    print("Preparing final train and test sets using pruned features...")
    data = prepare_data(df, feature_cols=kept_features)
    
    # Save Scaler
    scaler_path = os.path.join(models_dir, f"{ticker}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(data["scaler"], f)
    print(f"Scaler saved to {scaler_path}")
    
    # 1. Train Baseline Random Forest on pruned features
    print("Training final Baseline Random Forest model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(data["X_train_flat"], data["y_train_flat"])
    
    # Save RF Model
    rf_path = os.path.join(models_dir, f"{ticker}_rf.pkl")
    with open(rf_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"Baseline RF saved to {rf_path}")
    
    # 1.5. Train MLP Neural Network (Sequence-aware)
    print("Training MLP Neural Network model...")
    X_train_seq_flat = data["X_train_seq"].reshape(len(data["X_train_seq"]), -1)
    # Scale target returns by 100 to percentages for optimal neural network convergence
    y_train_seq_scaled = data["y_train_seq"] * 100.0
    mlp = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=400, alpha=0.01, early_stopping=True, random_state=42)
    mlp.fit(X_train_seq_flat, y_train_seq_scaled)
    
    # Save MLP Model
    mlp_path = os.path.join(models_dir, f"{ticker}_mlp.pkl")
    with open(mlp_path, 'wb') as f:
        pickle.dump(mlp, f)
    print(f"MLP Neural Network saved to {mlp_path}")
    
    if TF_AVAILABLE:
        # 2. Train LSTM Model
        print("Training LSTM model...")
        input_shape = (data["X_train_seq"].shape[1], data["X_train_seq"].shape[2])
        lstm = build_lstm_model(input_shape)
        
        # Define simple early stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        lstm.fit(
            data["X_train_seq"], data["y_train_seq"],
            validation_split=0.1,
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save LSTM Model
        lstm_path = os.path.join(models_dir, f"{ticker}_lstm.keras")
        lstm.save(lstm_path)
        print(f"LSTM model saved to {lstm_path}")
        
        # 3. Train Transformer Model
        print("Training Transformer model...")
        transformer = build_transformer_model(input_shape)
        transformer.fit(
            data["X_train_seq"], data["y_train_seq"],
            validation_split=0.1,
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save Transformer Model
        transformer_path = os.path.join(models_dir, f"{ticker}_transformer.keras")
        transformer.save(transformer_path)
        print(f"Transformer model saved to {transformer_path}")
    else:
        print("TensorFlow not installed. Skipping LSTM & Transformer training in models.py pipeline.")
    
    # Evaluate models on test set (reconstructing prices)
    print("\nEvaluating models on Test Set...")
    rf_pred_ret = rf.predict(data["X_test_flat"])
    
    X_test_seq_flat = data["X_test_seq"].reshape(len(data["X_test_seq"]), -1)
    # Unscale predictions by dividing by 100
    mlp_pred_ret = mlp.predict(X_test_seq_flat) / 100.0
    
    test_close = data["test_raw_close"]
    actual_prices = data["test_raw_target_close"]
    
    eval_list = [("Random Forest", rf_pred_ret), ("Neural Network (MLP)", mlp_pred_ret)]
    
    if TF_AVAILABLE:
        try:
            lstm_pred_ret = lstm.predict(data["X_test_seq"]).flatten()
            trans_pred_ret = transformer.predict(data["X_test_seq"]).flatten()
            eval_list.append(("LSTM", lstm_pred_ret))
            eval_list.append(("Transformer", trans_pred_ret))
        except Exception as e:
            print(f"Error evaluating deep learning models: {e}")
            
    for name, pred_ret in eval_list:
        pred_prices = test_close * (1 + pred_ret)
        mse = mean_squared_error(actual_prices, pred_prices)
        mae = mean_absolute_error(actual_prices, pred_prices)
        r2 = r2_score(actual_prices, pred_prices)
        print(f"{name} -> Price MSE: {mse:.4f}, Price MAE: {mae:.4f}, Price R2: {r2:.4f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate forecasting models")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--processed_file", type=str, default="data/processed/{ticker}_processed.csv", help="Template path to processed CSV file")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save models")
    
    args = parser.parse_args()
    
    train_pipeline(args.ticker, args.processed_file, args.models_dir)
