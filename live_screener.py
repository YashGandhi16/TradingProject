import os
import re
import pandas as pd
import yfinance as yf
import xgboost as xgb
from datetime import datetime

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_PATH = os.path.join(BASE_DIR, "ticker_list.md")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "xgb_model.json")
CSV_PATH = os.path.join(BASE_DIR, "data", "trades_scaled.csv")

def parse_watchlist(filepath):
    """Reads ticker_list.md and extracts clean ticker symbols."""
    tickers = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                clean_line = re.sub(r'[^A-Za-z]', '', line.strip().upper())
                if clean_line:
                    tickers.append(clean_line)
    except FileNotFoundError:
        print(f"[!] Error: {filepath} not found.")
        return []
    return list(set(tickers))

def get_human_checklist(ticker):
    """Prompts the user for the manual checklist items."""
    print(f"\n" + "="*50)
    print(f"🚨 SETUP DETECTED: {ticker} 🚨")
    print(f"Please review the chart for {ticker} and answer the following:")
    print("="*50)
    
    def ask_bool(prompt):
        while True:
            ans = input(prompt + " (y/n): ").lower().strip()
            if ans in ['y', 'yes']: return 1
            if ans in ['n', 'no']: return 0
            print("Please enter 'y' or 'n'.")

    return {
        '1W_Trend_Valid': ask_bool("Is the 1-Week Macro Trend Valid?"),
        '1D_Trend_Valid': ask_bool("Is the 1-Day Trend Valid?"),
        '1D_Volume_Contracting': ask_bool("Is 1D Volume Contracting?"),
        'Breakout_Open_Valid': ask_bool("Was the Breakout Open Valid?"),
        'Intraday_Volume_Spike': ask_bool("Is there an Intraday Volume Spike?"),
        'Consol_Close_Below_8EMA': ask_bool("Did consolidation close below 8 EMA?"),
        'Consol_Close_Below_21EMA': ask_bool("Did consolidation close below 21 EMA?"),
        'Consol_Close_Below_50EMA': ask_bool("Did consolidation close below 50 EMA?"),
        '>=6_days_consol': ask_bool("Is the consolidation >= 6 days?")
    }

def append_to_csv(trade_data, checklist):
    """Appends the new live trade to trades_scaled.csv with strict column ordering."""
    if not os.path.exists(CSV_PATH):
        print(f"[!] Warning: {CSV_PATH} not found. Cannot append trade.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Base dictionary for the row
    new_row = {
        'Trade_ID': trade_data['trade_id'],
        'Ticker': trade_data['ticker'],
        'Outcome': 'PENDING',
        'Date': trade_data['date']  # Placed directly after Outcome
    }
    
    # Add checklist values
    for k, v in checklist.items():
        new_row[k] = v
        
    # Create DataFrame and reindex to match the exact columns of the original CSV
    new_df = pd.DataFrame([new_row])
    new_df = new_df.reindex(columns=df.columns)
    
    # Append and save
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"[+] Successfully logged {trade_data['trade_id']} to trades_scaled.csv as PENDING.")
    
def scan_market():
    tickers = parse_watchlist(WATCHLIST_PATH)
    if not tickers:
        print("No tickers found to scan. Exiting.")
        return

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model file not found at {MODEL_PATH}. Run train_baseline.py first.")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    feature_names = model.get_booster().feature_names

    print(f"Starting Live Scan on {len(tickers)} tickers...")
    data = yf.download(tickers, period="3mo", interval="1d", group_by="ticker", progress=False)
    
    setups_found = 0
    
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = data.copy()
            else:
                df = data[ticker].copy()
                
            if df.empty:
                continue
                
            df.columns = df.columns.str.lower()
            df = df.dropna()
            
            # Math features calculation
            df['8ema'] = df['close'].ewm(span=8, adjust=False).mean()
            df['21ema'] = df['close'].ewm(span=21, adjust=False).mean()
            df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
            df['rolling_high_20'] = df['high'].rolling(20).max().shift(1)
            
            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            
            is_breakout = today['close'] > today['rolling_high_20']
            
            if is_breakout:
                setups_found += 1
                checklist = get_human_checklist(ticker)
                
                # Calculate basic math features to match training schema
                price_to_8ema_delta = (today['close'] - today['8ema']) / today['8ema']
                macro_breakout_gap_pct = (today['open'] - yesterday['close']) / yesterday['close']
                macro_vol_contraction_ratio = df['volume'].iloc[-5:].std() / df['volume'].iloc[-20:].std()
                intraday_rvol = today['volume'] / df['volume'].rolling(20).mean().iloc[-1]
                
                # Construct feature dictionary
                inference_data = {
                    'price_to_8ema_delta': price_to_8ema_delta,
                    'macro_breakout_gap_pct': macro_breakout_gap_pct,
                    'macro_vol_contraction_ratio': macro_vol_contraction_ratio,
                    'intraday_rvol': intraday_rvol,
                    'intraday_vol_contraction_ratio': 1.0,
                    'VCP_ATR_Ratio': 1.0,        # <-- ADD THIS PLACEHOLDER
                    'SPY_Trend_Valid': 1,        # <-- ADD THIS PLACEHOLDER
                    **checklist
                }
                
                # Ensure feature order matches the model
                X_inference = pd.DataFrame([inference_data])[feature_names]
                
                # Predict
                pred_proba = model.predict_proba(X_inference)[0][1]
                prediction = model.predict(X_inference)[0]
                
                print("\n" + "="*30)
                print(f"🤖 MODEL INFERENCE FOR {ticker}")
                print(f"Win Probability: {pred_proba * 100:.2f}%")
                if prediction == 1:
                    print("✅ RECOMMENDATION: TAKE TRADE (BUY)")
                else:
                    print("❌ RECOMMENDATION: SKIP (Value Trap)")
                print("="*30)
                
                # Log trade data
                current_date_str = datetime.now().strftime("%Y-%m-%d")
                trade_id = f"{ticker}_{datetime.now().strftime('%Y%m%d')}"
                
                trade_info = {
                    'trade_id': trade_id,
                    'ticker': ticker,
                    'date': current_date_str
                }
                
                append_to_csv(trade_info, checklist)
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    if setups_found == 0:
        print("\nScan complete. No mathematical breakouts detected today.")

if __name__ == "__main__":
    scan_market()