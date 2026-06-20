def get_consensus_signal(rf_pred, nn_pred, threshold=0.005):
    """
    rf_pred: Prediction from Random Forest (predicted return/movement)
    nn_pred: Prediction from Neural Network (predicted return/movement)
    threshold: Minimum movement to consider as a 'signal'
    """
    # Define directional bias (-1 for sell, 0 for hold, 1 for buy)
    signals = []
    for pred in [rf_pred, nn_pred]:
        if pred > threshold:
            signals.append(1)  # Buy
        elif pred < -threshold:
            signals.append(-1) # Sell
        else:
            signals.append(0)  # Hold
            
    # Consensus: Only trade if both models agree
    if signals[0] == signals[1] and signals[0] != 0:
        return signals[0] # Return the consensus signal
    else:
        return 0 # Stay in 'Hold' if they disagree

import argparse
import pandas as pd
import numpy as np

def run_backtest(dates, actual_prices, predicted_prices, predicted_prices_2=None, initial_capital=10000.0, buy_threshold=0.01, sell_threshold=0.01, transaction_fee=0.001):
    """
    Backtest trading strategy based on model predictions.
    
    Signals:
    - BUY: Predicted tomorrow's price is > today's price by buy_threshold%
    - SELL: Predicted tomorrow's price is < today's price by sell_threshold%
    - HOLD: Otherwise
    """
    capital = initial_capital
    position = 0.0  # number of shares held
    portfolio_values = []
    trades = []
    
    # State tracking
    has_position = False
    buy_price = 0.0
    
    for i in range(len(actual_prices) - 1):
        date = dates[i]
        today_price = actual_prices[i]
        
        if predicted_prices_2 is not None:
            # Consensus voting system
            rf_tomorrow = predicted_prices[i]
            nn_tomorrow = predicted_prices_2[i]
            rf_change = (rf_tomorrow - today_price) / today_price
            nn_change = (nn_tomorrow - today_price) / today_price
            
            signal = get_consensus_signal(rf_change, nn_change, buy_threshold)
        else:
            # Single model strategy
            predicted_tomorrow = predicted_prices[i]
            pred_change = (predicted_tomorrow - today_price) / today_price
            
            if pred_change > buy_threshold:
                signal = 1
            elif pred_change < -sell_threshold:
                signal = -1
            else:
                signal = 0
        
        # Execute signals
        if signal == 1 and not has_position:
            # Buy shares
            shares_to_buy = (capital * (1 - transaction_fee)) / today_price
            position = shares_to_buy
            capital = 0.0
            has_position = True
            buy_price = today_price
            trades.append({
                "date": date,
                "type": "BUY",
                "price": today_price,
                "shares": shares_to_buy,
                "cash": capital,
                "portfolio_value": position * today_price
            })
            print(f"[{date}] BUY {shares_to_buy:.2f} shares at {today_price:.2f}")
            
        elif signal == -1 and has_position:
            # Sell shares
            revenue = position * today_price * (1 - transaction_fee)
            capital = revenue
            position = 0.0
            has_position = False
            trades.append({
                "date": date,
                "type": "SELL",
                "price": today_price,
                "shares": 0.0,
                "cash": capital,
                "portfolio_value": capital
            })
            print(f"[{date}] SELL shares at {today_price:.2f}, Cash: {capital:.2f}")
            
        # Track portfolio value at the end of the day
        current_val = capital + (position * today_price)
        portfolio_values.append(current_val)
        
    # Append final portfolio value for the last day
    final_price = actual_prices[-1]
    final_val = capital + (position * final_price)
    portfolio_values.append(final_val)
    
    # Calculate performance metrics
    portfolio_values = np.array(portfolio_values)
    dates_list = list(dates)
    
    total_return = (final_val - initial_capital) / initial_capital
    buy_and_hold_return = (final_price - actual_prices[0]) / actual_prices[0]
    
    # Daily returns for Sharpe Ratio
    daily_returns = pd.Series(portfolio_values).pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
    else:
        sharpe_ratio = 0.0
        
    # Maximum Drawdown
    peaks = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - peaks) / peaks
    max_drawdown = drawdowns.min()
    
    # Win Rate
    # Calculate trades performance (buy to sell roundtrips)
    completed_trades = []
    for k in range(0, len(trades) - 1, 2):
        if trades[k]["type"] == "BUY" and k + 1 < len(trades) and trades[k+1]["type"] == "SELL":
            buy_trade = trades[k]
            sell_trade = trades[k+1]
            profit = (sell_trade["price"] - buy_trade["price"]) / buy_trade["price"]
            completed_trades.append(profit)
    
    win_rate = np.mean([1 if p > 0 else 0 for p in completed_trades]) if completed_trades else 0.0
    
    summary = {
        "initial_capital": initial_capital,
        "final_value": final_val,
        "total_return": total_return,
        "buy_and_hold_return": buy_and_hold_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "num_trades": len(trades),
        "completed_trades": completed_trades
    }
    
    df_portfolio = pd.DataFrame({
        "Date": dates_list,
        "Actual_Price": actual_prices,
        "Portfolio_Value": portfolio_values,
        "Buy_and_Hold_Value": (initial_capital / actual_prices[0]) * actual_prices
    })
    
    return df_portfolio, trades, summary

if __name__ == "__main__":
    # Test simple stub backtest
    print("Backtester script compiled successfully.")
