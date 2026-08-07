import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "trades_scaled.csv")

def log_historical_trade():
    print("="*60)
    print(" 📈 HISTORICAL DATA MINER (DATASET SCALING) 📈 ")
    print("="*60)
    print("Rapidly inject historical setups. Type 'quit' at any time to exit.\n")

    if not os.path.exists(CSV_PATH):
        print(f"[!] Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Ensure no unnamed pandas columns corrupt the injection
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    count = 0
    while True:
        ticker = input("\nTicker (e.g., TSLA): ").strip().upper()
        if ticker.lower() == 'quit':
            break
        if not ticker:
            continue
            
        date_str = input(f"Entry Date for {ticker} (YYYY-MM-DD): ").strip()
        if date_str.lower() == 'quit':
            break
            
        # Automatically append a default time if only the date is provided
        if len(date_str) == 10:
            date_str += " 09:45:00"
        elif len(date_str) == 16:
            date_str += ":00"
            
        try:
            # Validate the datetime format
            dt_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("[!] Invalid format. Please use YYYY-MM-DD.")
            continue

        outcome_input = input("Outcome (W = Win, L = Loss): ").strip().upper()
        if outcome_input.lower() == 'quit':
            break
            
        if outcome_input == 'W':
            outcome = 'Win'
        elif outcome_input == 'L':
            outcome = 'Loss'
        else:
            print("[!] Invalid outcome. You must type 'W' or 'L'.")
            continue

        # Append _HIST so you always know which trades were artificially injected
        trade_id = f"{ticker}_{dt_obj.strftime('%Y%m%d')}"

        new_row = {
            'Trade_ID': trade_id,
            'Ticker': ticker,
            'Outcome': outcome,
            'Entry_Time': date_str,  # Matches the format your download script needs
            'Date': dt_obj.strftime('%Y-%m-%d')
        }

        new_df = pd.DataFrame([new_row])
        # Reindexing strictly aligns the new row with the existing CSV architecture
        new_df = new_df.reindex(columns=df.columns)
        
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        
        count += 1
        print(f"[+] Injected: {trade_id} logged as {outcome.upper()}.")

    print("=" * 60)
    print(f"[+] Mining session complete. {count} historical trades added.")
    print("-> NEXT STEP 1: Run 'python3 download_data.py' to fetch the local CSVs.")
    print("-> NEXT STEP 2: Run 'python3 src/models/run_pipeline.py' to calculate the math.")
    print("-> NEXT STEP 3: Run 'python3 src/models/train_robust.py' to update the model.")

if __name__ == "__main__":
    log_historical_trade()