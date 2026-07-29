import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "trades_scaled.csv")

def log_trade():
    ticker = input("\nEnter the ticker for the executed trade: ").strip().upper()
    if not ticker:
        print("No ticker entered. Exiting.")
        return

    if not os.path.exists(CSV_PATH):
        print(f"[!] Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    date_str = datetime.now().strftime("%Y-%m-%d")
    trade_id = f"{ticker}_{datetime.now().strftime('%Y%m%d')}"

    # Set up the base information
    new_row = {
        'Trade_ID': trade_id,
        'Ticker': ticker,
        'Outcome': 'PENDING',
        'Date': date_str
    }

    # Create a new DataFrame for the single row
    new_df = pd.DataFrame([new_row])
    # Aligning columns leaves all the math features (ATR, EMAs) as NaN
    new_df = new_df.reindex(columns=df.columns)
    
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    
    print("\n" + "="*40)
    print(f"[+] TRADE LOGGED SUCCESSFULLY")
    print(f"    ID:      {trade_id}")
    print(f"    Ticker:  {ticker}")
    print(f"    Status:  PENDING")
    print("="*40)
    print("-> Note: Run `python3 backfill_features.py` tonight.")
    print("   This will automatically calculate and fill in the missing")
    print("   mathematical metrics for this row before your next pipeline run.")

if __name__ == "__main__":
    log_trade()