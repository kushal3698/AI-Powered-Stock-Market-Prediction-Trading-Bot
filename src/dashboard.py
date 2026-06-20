import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# TensorFlow is optional - loaded lazily when models exist
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None

# Import project modules (TF-independent ones always)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingest import ingest_data
from preprocessing import preprocess_pipeline
from sentiment import analyze_sentiment
from backtest import run_backtest

FEATURE_COLS = [
    'Close', 'EMA_10', 'EMA_50',
    'RSI_14', 'MACD', 'MACD_Signal',
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'ATR_14', 'Daily_Return',
    'News_Sentiment'
]

def _prepare_data(df, seq_length=10, test_split=0.2):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import pickle
    import os
    
    # Try to load kept features list dynamically from models
    kept_features_path = f"models/{ticker}_kept_features.pkl"
    if os.path.exists(kept_features_path):
        with open(kept_features_path, 'rb') as f:
            feature_cols = pickle.load(f)
    else:
        feature_cols = FEATURE_COLS
        
    all_cols = feature_cols + ['Target_Return']
    df_clean = df.dropna(subset=all_cols).reset_index(drop=True)
    split_idx = int(len(df_clean) * (1 - test_split))
    train_df = df_clean.iloc[:split_idx]
    test_df  = df_clean.iloc[split_idx:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_df[feature_cols])
    X_test_s  = scaler.transform(test_df[feature_cols])
    
    train_s = pd.DataFrame(X_train_s, columns=feature_cols)
    train_s['Target_Return'] = train_df['Target_Return'].values
    
    test_s  = pd.DataFrame(X_test_s, columns=feature_cols)
    test_s['Target_Return'] = test_df['Target_Return'].values

    def make_seq(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[feature_cols].iloc[i:i+seq_len].values)
            y.append(data['Target_Return'].iloc[i+seq_len-1])
        return np.array(X), np.array(y)

    X_tr, y_tr = make_seq(train_s, seq_length)
    X_te, y_te = make_seq(test_s,  seq_length)
    
    test_raw_dates = df_clean['Date'].iloc[split_idx + seq_length - 1 : split_idx + seq_length - 1 + len(y_te)].values
    test_raw_close = df_clean['Close'].iloc[split_idx + seq_length - 1 : split_idx + seq_length - 1 + len(y_te)].values
    test_raw_target_close = df_clean['Target_Close'].iloc[split_idx + seq_length - 1 : split_idx + seq_length - 1 + len(y_te)].values
    
    return {
        "X_train_flat": X_tr[:, -1, :], "y_train_flat": y_tr,
        "X_test_flat":  X_te[:, -1, :], "y_test_flat":  y_te,
        "X_train_seq":  X_tr, "y_train_seq": y_tr,
        "X_test_seq":   X_te, "y_test_seq":  y_te,
        "scaler": scaler,
        "test_raw_dates": test_raw_dates,
        "test_raw_close": test_raw_close,
        "test_raw_target_close": test_raw_target_close
    }

# Set page config for a premium dark look
st.set_page_config(
    page_title="AI-Powered Stock Market Prediction & Trading Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .reportview-container {
        background: #0f1116;
    }
    .stMetric {
        background-color: #1e222b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3440;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 AI-Powered Stock Market Prediction & Trading Bot")
st.subheader("A modular Python assistant for predictive analysis, sentiment tracking, and backtesting")

# Sidebar Configuration
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

# Pipeline status
raw_file = f"data/raw/{ticker}.csv"
processed_file = f"data/processed/{ticker}_processed.csv"
model_prefix = f"models/{ticker}"

st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline Actions")

# Buttons in Sidebar
if st.sidebar.button("🚀 Ingest & Preprocess Data"):
    with st.spinner("Ingesting from yFinance..."):
        try:
            ingest_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            preprocess_pipeline(ticker)
            st.sidebar.success("Ingestion & Preprocessing Completed!")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

if st.sidebar.button("⚙️ Train Models & SHAP"):
    with st.spinner("Training models..."):
        try:
            import pickle, numpy as np
            import shap, matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Call central training pipeline in models.py (handles RF, MLP, etc.)
            from models import train_pipeline
            train_pipeline(ticker)
            
            # Load kept features to align columns
            with open(f"models/{ticker}_kept_features.pkl", 'rb') as f:
                kept_features = pickle.load(f)
                
            # Load final RF model
            with open(f"{model_prefix}_rf.pkl", 'rb') as f:
                rf_tmp = pickle.load(f)
                
            # Prepare dataset with updated features list
            df_tmp = pd.read_csv(processed_file)
            data_tmp = _prepare_data(df_tmp)
            
            # Generate SHAP explanations for the final RF model
            X_s = pd.DataFrame(data_tmp["X_test_flat"][:200], columns=kept_features)
            expl = shap.TreeExplainer(rf_tmp)
            sv   = expl.shap_values(X_s)
            fig, _ = plt.subplots(figsize=(10,6))
            shap.summary_plot(sv, X_s, feature_names=kept_features, show=False)
            plt.tight_layout()
            plt.savefig(f"{model_prefix}_shap_summary.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            st.sidebar.success("Pipeline Training & SHAP summary completed!")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# Data validation checks
data_exists = os.path.exists(processed_file)
models_exist = os.path.exists(f"{model_prefix}_rf.pkl")

if not data_exists:
    st.warning("⚠️ No processed data found. Please click **Ingest & Preprocess Data** in the sidebar to download stock details.")
else:
    # Load processed data
    df = pd.read_csv(processed_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # ------------------ MAIN TABS ------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Explorer", 
        "🧠 AI Forecasting", 
        "📰 Sentiment Center", 
        "⚖️ Backtesting Arena",
        "🔍 Model Explainability"
    ])
    
    # --- TAB 1: Market Explorer ---
    with tab1:
        st.header("Stock Market Explorer")
        st.caption(f"Historical price data for {ticker}")
        
        # Interactive Plotly Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlestick"
        ), row=1, col=1)
        
        # Technical overlays checkboxes
        col_sma10, col_sma50, col_bb = st.columns(3)
        with col_sma10:
            show_sma10 = st.checkbox("Show SMA 10", value=True)
        with col_sma50:
            show_sma50 = st.checkbox("Show SMA 50", value=False)
        with col_bb:
            show_bb = st.checkbox("Show Bollinger Bands (20)", value=False)
            
        if show_sma10:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_10'], name='SMA 10', line=dict(color='orange', width=1.5)), row=1, col=1)
        if show_sma50:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1.5)), row=1, col=1)
        if show_bb:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
            
        # Volume
        fig.add_trace(go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name="Volume",
            marker_color='lightblue'
        ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent data table
        st.subheader("Recent Historical Data")
        st.dataframe(df.tail(10).style.format({
            "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Close": "{:.2f}",
            "Daily_Return": "{:.2%}", "RSI_14": "{:.1f}"
        }))
        
    # --- TAB 2: AI Forecasting ---
    with tab2:
        st.header("AI Forecasting & Predictions")
        
        if not models_exist:
            st.info("💡 Forecasting models are not trained yet. Please click **Train Models & SHAP** in the sidebar.")
        else:
            with st.spinner("Loading models and generating forecasts..."):
                # Load models
                with open(f"{model_prefix}_rf.pkl", 'rb') as f:
                    rf_model = pickle.load(f)
                with open(f"{model_prefix}_scaler.pkl", 'rb') as f:
                    scaler = pickle.load(f)
                    
                lstm_model = keras.models.load_model(f"{model_prefix}_lstm.keras") if TF_AVAILABLE else None
                trans_model = keras.models.load_model(f"{model_prefix}_transformer.keras") if TF_AVAILABLE else None
                
                # Fetch actual real-time sentiment score
                try:
                    sentiment_data = analyze_sentiment(ticker)
                    real_time_sentiment = sentiment_data['average_sentiment']
                except Exception as e:
                    real_time_sentiment = 0.0

                # Inject real-time sentiment into the latest row of df
                df_sent = df.copy()
                if 'News_Sentiment' in df_sent.columns and len(df_sent) > 0:
                    df_sent.loc[df_sent.index[-1], 'News_Sentiment'] = real_time_sentiment

                # Prepare forecasting dataset with real-time sentiment injected
                data = _prepare_data(df_sent)
                
                # Predictions - only RF guaranteed, LSTM/Transformer need TF
                import numpy as np
                rf_pred_ret = rf_model.predict(data["X_test_flat"])
                
                # Load MLP Model
                mlp_model = None
                if os.path.exists(f"{model_prefix}_mlp.pkl"):
                    with open(f"{model_prefix}_mlp.pkl", 'rb') as f:
                        mlp_model = pickle.load(f)
                
                if mlp_model is not None:
                    X_test_seq_flat = data["X_test_seq"].reshape(len(data["X_test_seq"]), -1)
                    mlp_pred_ret = mlp_model.predict(X_test_seq_flat) / 100.0
                else:
                    mlp_pred_ret = np.full(len(rf_pred_ret), np.nan)
                    
                lstm_pred_ret  = lstm_model.predict(data["X_test_seq"]).flatten()  if (TF_AVAILABLE and lstm_model  is not None) else np.full(len(rf_pred_ret), np.nan)
                trans_pred_ret = trans_model.predict(data["X_test_seq"]).flatten() if (TF_AVAILABLE and trans_model is not None) else np.full(len(rf_pred_ret), np.nan)
                
                # Display metrics
                test_dates = data["test_raw_dates"]
                test_close = data["test_raw_close"]
                actual_target_close = data["test_raw_target_close"]
                
                # Reconstruct prices from returns
                rf_pred = test_close * (1 + rf_pred_ret)
                mlp_pred = test_close * (1 + mlp_pred_ret) if not np.all(np.isnan(mlp_pred_ret)) else np.full(len(rf_pred_ret), np.nan)
                lstm_pred = test_close * (1 + lstm_pred_ret) if not np.all(np.isnan(lstm_pred_ret)) else np.full(len(rf_pred_ret), np.nan)
                trans_pred = test_close * (1 + trans_pred_ret) if not np.all(np.isnan(trans_pred_ret)) else np.full(len(rf_pred_ret), np.nan)
                
                st.subheader("Model Validation Performance")
                col_rf, col_mlp, col_lstm, col_trans = st.columns(4)
                
                from sklearn.metrics import mean_absolute_error, r2_score
                for col, name, pred in [(col_rf, "Random Forest (Baseline)", rf_pred), 
                                        (col_mlp, "Neural Network (MLP)", mlp_pred),
                                        (col_lstm, "LSTM", lstm_pred), 
                                        (col_trans, "Transformer", trans_pred)]:
                    with col:
                        if np.all(np.isnan(pred)):
                            st.metric(label=name, value="Unavailable", delta="Install TensorFlow")
                        else:
                            mae = mean_absolute_error(actual_target_close, pred)
                            r2 = r2_score(actual_target_close, pred)
                            st.metric(label=name, value=f"MAE: {mae:.2f}", delta=f"R2: {r2:.2f}")
                
                # Forecast vs Actual Chart
                st.subheader("Test Set Forecast vs. Actual Close Prices")
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=test_dates, y=actual_target_close, name="Actual Close", line=dict(color="white", width=2)))
                fig_forecast.add_trace(go.Scatter(x=test_dates, y=rf_pred, name="Random Forest", line=dict(color="orange", width=1.5)))
                if not np.all(np.isnan(mlp_pred)):
                    fig_forecast.add_trace(go.Scatter(x=test_dates, y=mlp_pred, name="Neural Network (MLP)", line=dict(color="lightgreen", width=1.5)))
                if not np.all(np.isnan(lstm_pred)):
                    fig_forecast.add_trace(go.Scatter(x=test_dates, y=lstm_pred, name="LSTM", line=dict(color="cyan", width=1.5)))
                if not np.all(np.isnan(trans_pred)):
                    fig_forecast.add_trace(go.Scatter(x=test_dates, y=trans_pred, name="Transformer", line=dict(color="magenta", width=1.5)))
                
                fig_forecast.update_layout(
                    height=500,
                    template="plotly_dark",
                    margin=dict(t=20, b=20, l=20, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Next Day Recommendation
                st.subheader("AI Trading Recommendation (Next Session)")
                
                # Take latest sequence for prediction
                latest_seq = data["X_test_seq"][-1:] # last sequence in test set
                latest_flat = data["X_test_flat"][-1:]
                current_price = test_close[-1]
                
                rf_next_ret = rf_model.predict(latest_flat)[0]
                lstm_next_ret  = lstm_model.predict(latest_seq)[0][0]  if (TF_AVAILABLE and lstm_model  is not None) else None
                trans_next_ret = trans_model.predict(latest_seq)[0][0] if (TF_AVAILABLE and trans_model is not None) else None
                
                latest_seq_flat = latest_seq.reshape(1, -1)
                mlp_next_ret = (mlp_model.predict(latest_seq_flat)[0] / 100.0) if mlp_model is not None else None
                
                rf_next = current_price * (1 + rf_next_ret)
                mlp_next = current_price * (1 + mlp_next_ret) if mlp_next_ret is not None else None
                lstm_next = current_price * (1 + lstm_next_ret) if lstm_next_ret is not None else None
                trans_next = current_price * (1 + trans_next_ret) if trans_next_ret is not None else None
                
                # Import consensus signal from backtest module
                from backtest import get_consensus_signal
                # Expected returns are passed to consensus logic (buy/sell threshold defaults to 0.005)
                consensus_signal = get_consensus_signal(rf_next_ret, mlp_next_ret, threshold=0.005)

                col_rf_rec, col_mlp_rec, col_cons_rec, col_lstm_rec, col_trans_rec = st.columns(5)
                
                # Render Consensus Card separately
                with col_cons_rec:
                    if consensus_signal == 1:
                        rec = "BUY"
                        rec_color = "green"
                    elif consensus_signal == -1:
                        rec = "SELL"
                        rec_color = "red"
                    else:
                        rec = "HOLD"
                        rec_color = "gray"
                        
                    st.markdown(f"""
                        <div style='background-color: #1e222b; padding: 20px; border-radius: 10px; border-left: 5px solid {rec_color}; height: 160px;'>
                            <h4>Consensus (RF+MLP)</h4>
                            <p style='font-size: 22px; font-weight: bold; margin-top: 10px; color: {rec_color};'>{rec}</p>
                            <p style='color: #888; font-size: 12px; margin: 0;'>Only triggers if both models agree on direction.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                for col, name, next_val in [(col_rf_rec, "Random Forest", rf_next),
                                            (col_mlp_rec, "Neural Network (MLP)", mlp_next),
                                            (col_lstm_rec, "LSTM", lstm_next),
                                            (col_trans_rec, "Transformer", trans_next)]:
                    with col:
                        if next_val is None:
                            st.markdown(f"""
                                <div style='background-color: #1e222b; padding: 20px; border-radius: 10px; border-left: 5px solid #555;'>
                                    <h4>{name}</h4>
                                    <p style='color: #888; font-size: 14px;'>TensorFlow not installed.<br>Install to enable this model.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            pct_change = (next_val - current_price) / current_price
                            if pct_change > 0.01:
                                rec = "BUY"
                                rec_color = "green"
                            elif pct_change < -0.01:
                                rec = "SELL"
                                rec_color = "red"
                            else:
                                rec = "HOLD"
                                rec_color = "gray"
                            st.markdown(f"""
                                <div style='background-color: #1e222b; padding: 20px; border-radius: 10px; border-left: 5px solid {rec_color};'>
                                    <h4>{name} Forecast</h4>
                                    <p style='font-size: 24px; font-weight: bold; margin: 0;'>${next_val:.2f}</p>
                                    <p style='color: {rec_color}; font-weight: bold; font-size: 18px;'>{rec} ({pct_change:+.2%})</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
    # --- TAB 3: Sentiment Center ---
    with tab3:
        st.header("Financial News Sentiment Analysis")
        st.caption(f"Scraping current news signals for {ticker}")
        
        with st.spinner("Fetching and analyzing sentiment..."):
            sentiment_data = analyze_sentiment(ticker)
            
            # Show summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Overall Sentiment Indicator", value=f"{sentiment_data['average_sentiment']:.2f}", delta=sentiment_data['sentiment_label'])
            with col2:
                st.metric(label="Positive Articles", value=sentiment_data['stats']['positive'])
            with col3:
                st.metric(label="Neutral Articles", value=sentiment_data['stats']['neutral'])
            with col4:
                st.metric(label="Negative Articles", value=sentiment_data['stats']['negative'])
                
            # Render sentiment breakdown chart
            st.subheader("Sentiment Distribution")
            fig_sent = go.Figure(data=[go.Bar(
                x=['Positive', 'Neutral', 'Negative'],
                y=[sentiment_data['stats']['positive'], sentiment_data['stats']['neutral'], sentiment_data['stats']['negative']],
                marker_color=['green', 'gray', 'red']
            )])
            fig_sent.update_layout(height=300, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_sent, use_container_width=True)
            
            # Render News Table
            st.subheader("Recent News Headlines")
            if sentiment_data['articles']:
                articles_df = pd.DataFrame(sentiment_data['articles'])
                # Format publish time
                articles_df['Published At'] = pd.to_datetime(articles_df['pub_time'], unit='s')
                st.dataframe(articles_df[['title', 'publisher', 'sentiment_score', 'sentiment_label', 'Published At']].style.format({
                    "sentiment_score": "{:.2f}"
                }))
            else:
                st.write("No news articles found.")
                
    # --- TAB 4: Backtesting Arena ---
    with tab4:
        st.header("Trading Strategy Backtester")
        st.caption("Evaluate your model's profit-generating capability on historical test data")
        
        if not models_exist:
            st.info("💡 Please train models first to perform backtesting.")
        else:
            # Backtest settings
            col_cap, col_fee, col_th = st.columns(3)
            with col_cap:
                initial_capital = st.number_input("Starting Capital ($)", value=10000.0, step=1000.0)
            with col_fee:
                fee = st.number_input("Transaction Fee (%)", value=0.1, step=0.05) / 100.0
            with col_th:
                threshold = st.number_input("Trade Signal Threshold (%)", value=1.0, step=0.2) / 100.0
                
            model_choice = st.selectbox("Select Prediction Model for Trading Signals", ["Random Forest", "Neural Network (MLP)", "Consensus Voting (RF + MLP)", "LSTM", "Transformer"])
            
            # Load selected predictions
            data = _prepare_data(df)
            test_dates = data["test_raw_dates"]
            test_close = data["test_raw_close"]
            
            if model_choice == "Random Forest":
                with open(f"{model_prefix}_rf.pkl", 'rb') as f:
                    rf_model = pickle.load(f)
                pred_ret = rf_model.predict(data["X_test_flat"])
                pred_prices = test_close * (1 + pred_ret)
                pred_prices_2 = None
            elif model_choice == "Neural Network (MLP)":
                with open(f"{model_prefix}_mlp.pkl", 'rb') as f:
                    mlp_model = pickle.load(f)
                X_test_seq_flat = data["X_test_seq"].reshape(len(data["X_test_seq"]), -1)
                pred_ret = mlp_model.predict(X_test_seq_flat) / 100.0
                pred_prices = test_close * (1 + pred_ret)
                pred_prices_2 = None
            elif model_choice == "Consensus Voting (RF + MLP)":
                with open(f"{model_prefix}_rf.pkl", 'rb') as f:
                    rf_model = pickle.load(f)
                with open(f"{model_prefix}_mlp.pkl", 'rb') as f:
                    mlp_model = pickle.load(f)
                rf_pred_ret = rf_model.predict(data["X_test_flat"])
                X_test_seq_flat = data["X_test_seq"].reshape(len(data["X_test_seq"]), -1)
                mlp_pred_ret = mlp_model.predict(X_test_seq_flat) / 100.0
                pred_prices = test_close * (1 + rf_pred_ret)
                pred_prices_2 = test_close * (1 + mlp_pred_ret)
            elif model_choice == "LSTM":
                if not TF_AVAILABLE:
                    st.warning("TensorFlow not installed. Using Random Forest instead.")
                    with open(f"{model_prefix}_rf.pkl", 'rb') as f:
                        rf_model = pickle.load(f)
                    pred_ret = rf_model.predict(data["X_test_flat"])
                    pred_prices = test_close * (1 + pred_ret)
                else:
                    lstm_model = keras.models.load_model(f"{model_prefix}_lstm.keras")
                    pred_ret = lstm_model.predict(data["X_test_seq"]).flatten()
                    pred_prices = test_close * (1 + pred_ret)
            else:
                if not TF_AVAILABLE:
                    st.warning("TensorFlow not installed. Using Random Forest instead.")
                    with open(f"{model_prefix}_rf.pkl", 'rb') as f:
                        rf_model = pickle.load(f)
                    pred_ret = rf_model.predict(data["X_test_flat"])
                    pred_prices = test_close * (1 + pred_ret)
                else:
                    trans_model = keras.models.load_model(f"{model_prefix}_transformer.keras")
                    pred_ret = trans_model.predict(data["X_test_seq"]).flatten()
                    pred_prices = test_close * (1 + pred_ret)
                
            # Run simulation
            df_portfolio, trades, summary = run_backtest(
                dates=test_dates,
                actual_prices=test_close,
                predicted_prices=pred_prices,
                predicted_prices_2=pred_prices_2,
                initial_capital=initial_capital,
                buy_threshold=threshold,
                sell_threshold=threshold,
                transaction_fee=fee
            )
            
            # Summary Metrics Cards
            st.subheader("Simulation Results")
            col_ret, col_bh, col_sr, col_dd = st.columns(4)
            with col_ret:
                st.metric(label="Strategy Return", value=f"{summary['total_return']:.2%}", delta=f"${summary['final_value'] - initial_capital:+,.2f}")
            with col_bh:
                st.metric(label="Buy & Hold Return", value=f"{summary['buy_and_hold_return']:.2%}")
            with col_sr:
                st.metric(label="Sharpe Ratio", value=f"{summary['sharpe_ratio']:.2f}")
            with col_dd:
                st.metric(label="Max Drawdown", value=f"{summary['max_drawdown']:.2%}")
                
            # Portfolio Chart
            fig_port = go.Figure()
            fig_port.add_trace(go.Scatter(x=df_portfolio['Date'], y=df_portfolio['Portfolio_Value'], name='Strategy Portfolio', line=dict(color='green', width=2.5)))
            fig_port.add_trace(go.Scatter(x=df_portfolio['Date'], y=df_portfolio['Buy_and_Hold_Value'], name='Buy & Hold Benchmark', line=dict(color='white', width=1.5, dash='dash')))
            
            fig_port.update_layout(
                height=450,
                template="plotly_dark",
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_port, use_container_width=True)
            
            # Trades Table
            st.subheader("Transaction Log")
            if trades:
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df.style.format({
                    "price": "{:.2f}", "shares": "{:.2f}", "cash": "{:.2f}", "portfolio_value": "{:.2f}"
                }))
            else:
                st.write("No trades executed. Try lowering the Trade Signal Threshold.")
                
    # --- TAB 5: Model Explainability ---
    with tab5:
        st.header("Model Explainability (SHAP)")
        st.caption("Understand how different features/technical indicators influence baseline model predictions")
        
        shap_img = f"{model_prefix}_shap_summary.png"
        
        if not os.path.exists(shap_img):
            st.info("💡 SHAP explanation plot is not generated. Train the models in the sidebar to create SHAP summaries.")
        else:
            col_plot, col_info = st.columns([0.65, 0.35])
            with col_plot:
                st.image(shap_img, caption="SHAP Summary Plot (Random Forest Regressor)", use_container_width=True)
            with col_info:
                st.markdown("""
                ### How to read this plot?
                
                - **Feature Importance**: Features are ranked from top to bottom based on their average predictive influence. The top features have the strongest effect on price forecasting.
                - **Feature Value (Color)**: 
                  - **Red** indicates high feature value (e.g. high MACD or high Close price).
                  - **Blue** indicates low feature value (e.g. low Close price).
                - **SHAP Value (X-Axis)**: 
                  - A positive SHAP value (shifted right) means that feature value drives the prediction **higher** (bullish impact).
                  - A negative SHAP value (shifted left) means that feature value drives the prediction **lower** (bearish impact).
                  
                *This summary uses the latest 200 data points to calculate feature attributions.*
                """)
