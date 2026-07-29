import os
import pandas as pd
from src.features.build_features import build_macro_features, build_intraday_features
from src.features.merge_features import align_timeframes
from src.features.generate_labels import generate_targets
from src.data.split_and_store import save_and_register_dataset

def main():
    print("--- Starting Batch Feature Pipeline ---")

    # 1. Load the master registry from your trades_scaled.csv file
    # Check both names just in case
    trades_path = "data/trades_scaled.csv"
    if not os.path.exists(trades_path):
        trades_path = "data/trades.csv"
        
    if not os.path.exists(trades_path):
        print(f"Error: Could not find {trades_path}")
        return
        
    trades_df = pd.read_csv(trades_path)
    print(f"Found {len(trades_df)} trades to process in {os.path.basename(trades_path)}")

    all_processed_data = []
    
    # Define the CSV features we want to inject into the dataset
    csv_features = [
        '1W_Trend_Valid', '1D_Trend_Valid', '1D_Volume_Contracting',
        'Breakout_Open_Valid', 'Intraday_Volume_Spike',
        'Consol_Close_Below_8EMA', 'Consol_Close_Below_21EMA',
        'Consol_Close_Below_50EMA', '>=6_days_consol',
        'VCP_ATR_Ratio', 'SPY_Trend_Valid'
    ]

    # 2. Iterate over every trade in the CSV
    for index, row in trades_df.iterrows():
        # Handle Trade_ID safely
        if 'Trade_ID' in row and pd.notna(row['Trade_ID']):
            trade_id = str(row['Trade_ID']).strip()
        else:
            # Fallback if Trade_ID is missing
            trade_id = f"{row['Ticker']}_{index}"
            
        ticker = str(row['Ticker']).strip()
        print(f"Processing {trade_id} ({ticker})...")
        
        base_path = f"data/raw_dataset/{trade_id}"
        daily_path = f"{base_path}/daily.csv"
        intraday_path = f"{base_path}/intraday_5m.csv"
        
        # Safety check: ensure the folder and files actually exist before loading
        if not os.path.exists(daily_path) or not os.path.exists(intraday_path):
            print(f"  [!] Skipping {trade_id}: Missing raw data CSVs in {base_path}. Did you run your download script?")
            continue
            
        # 3. Load the raw data
        daily_df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        intraday_df = pd.read_csv(intraday_path, index_col=0, parse_dates=True)
        
        # Strip timezones
        daily_df.index = daily_df.index.tz_localize(None)
        intraday_df.index = intraday_df.index.tz_localize(None)
        
        # 4. Build Mathematical Features
        daily_features = build_macro_features(daily_df)
        intraday_features = build_intraday_features(intraday_df)
        
        # 5. Safely merge without forward-looking leakage
        merged_df = align_timeframes(intraday_df=intraday_features, macro_df=daily_features)
        
        # 6. Generate the classification targets
        final_trade_df = generate_targets(
            df=merged_df, 
            take_profit=0.10,   
            stop_loss=0.05,     
            hold_time_bars=1170 
        )
        
        # --- ISOLATE THE EXACT BREAKOUT BAR ---
        # The new CSV uses 'Date', the old used 'Entry_Time'. Let's check both.
        timestamp_col = 'Date' if 'Date' in row else 'Entry_Time'
        
        if timestamp_col in row and pd.notna(row[timestamp_col]):
            entry_time = pd.to_datetime(str(row[timestamp_col]).strip())
            
            # Find the exact match or nearest preceding 5m bar
            if entry_time in final_trade_df.index:
                final_trade_df = final_trade_df.loc[[entry_time]].copy() # ADDED .copy()
            else:
                # If exact minute isn't found, find the closest bar immediately before it
                past_bars = final_trade_df[final_trade_df.index <= entry_time]
                if not past_bars.empty:
                    final_trade_df = past_bars.tail(1).copy() # ADDED .copy()
                else:
                    final_trade_df = pd.DataFrame()
        else:
            # Only use math logic if human timestamp is totally missing
            final_trade_df['rolling_high_20'] = final_trade_df['high'].rolling(20).max().shift(1)
            breakout_mask = final_trade_df['close'] > final_trade_df['rolling_high_20']
            final_trade_df = final_trade_df[breakout_mask].head(1).copy() # ADDED .copy()
            
        if final_trade_df.empty:
            print(f"  [!] Skipping {trade_id}: Could not match your timestamp to the 5-min dataset.")
            continue
            
        # --- INJECT HUMAN LABELS & FEATURES ---
        # 1. Metadata
        final_trade_df.loc[:, 'ticker'] = ticker
        final_trade_df.loc[:, 'trade_id'] = trade_id
        
        # 2. Checklist & Math Features
        for col in csv_features:
            if col in row:
                val = row[col]
                if isinstance(val, bool) or str(val).upper() in ['TRUE', 'FALSE']:
                    final_trade_df.loc[:, col] = 1.0 if str(val).upper() == 'TRUE' else 0.0
                else:
                    final_trade_df.loc[:, col] = float(val)
                
        # 3. Ground Truth Override (CRITICAL)
        if 'Outcome' in row and pd.notna(row['Outcome']):
            outcome = str(row['Outcome']).strip().lower()
            if 'win' in outcome:
                final_trade_df.loc[:, 'target'] = 1
            elif 'loss' in outcome or 'avoid' in outcome:
                final_trade_df.loc[:, 'target'] = 0
        
        # Add to our master list
        all_processed_data.append(final_trade_df)

    # 7. Combine everything into one massive dataset
    if not all_processed_data:
        print("No valid trade folders were found. Exiting pipeline.")
        return
        
    print("\nConcatenating all processed trades into a master dataset...")
    master_df = pd.concat(all_processed_data)
    master_df = master_df.sort_index()

    # 8. Split chronologically, save to Parquet, and log to SQLite
    print("Saving datasets and registering metadata...")
    save_and_register_dataset(
        df=master_df,
        ticker="ALL_TRADES_HYBRID", 
        dataset_version="v2_human_verified"
    )

    print("--- Pipeline Complete! ---")

if __name__ == "__main__":
    main()