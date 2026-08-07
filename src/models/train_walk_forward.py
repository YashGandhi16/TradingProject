import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
import os

def train_walk_forward():
    print("="*60)
    print(" 📈 TIME-SERIES WALK-FORWARD VALIDATION 📈 ")
    print("="*60)
    
    # 1. Dynamically find the latest dataset
    processed_dir = "data/processed/"
    prefix = "ALL_TRADES_HYBRID"
    all_folders = [
        os.path.join(processed_dir, d) for d in os.listdir(processed_dir) 
        if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith(prefix)
    ]
    
    if not all_folders:
        print("[!] Error: No processed datasets found.")
        return

    latest_folder = max(all_folders, key=os.path.getmtime)
    print(f"[*] Dynamically loaded: {os.path.basename(latest_folder)}\n")
    
    # 2. Load and merge Train/Test to sort chronologically
    train = pd.read_parquet(os.path.join(latest_folder, "train.parquet"))
    test = pd.read_parquet(os.path.join(latest_folder, "test.parquet"))
    
    df = pd.concat([train, test]).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Total Chronological Dataset: {len(df)} rows")

    features = [
        'VCP_ATR_Ratio', '1D_Trend_Valid', 'Consol_Close_Below_50EMA', 
        'Breakout_Open_Valid', '1D_Volume_Contracting', 'intraday_rvol', 
        'macro_vol_contraction_ratio', 'macro_breakout_gap_pct', 
        'price_to_8ema_delta', '1W_Trend_Valid', 'intraday_vol_contraction_ratio', 
        'Intraday_Volume_Spike', 'Consol_Close_Below_8EMA', 
        'Consol_Close_Below_21EMA', '>=6_days_consol', 'SPY_Trend_Valid', 
        'Sector_Trend_Valid'
    ]

    X = df[features]
    y = df['target']

    # 3. Institutional Walk-Forward Splitter (4 Folds)
    tscv = TimeSeriesSplit(n_splits=4)
    
    model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, 
        random_state=42, eval_metric='logloss'
    )

    fold = 1
    total_trades = 0
    total_wins = 0

    print("\n--- Walk-Forward Results ---")
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train on the past
        model.fit(X_train, y_train)
        
        # Predict the future
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)
        
        # Calculate Fold Metrics
        trades_taken = sum(preds)
        wins = sum((preds == 1) & (y_test == 1))
        win_rate = (wins / trades_taken * 100) if trades_taken > 0 else 0.0
        
        total_trades += trades_taken
        total_wins += wins
        
        print(f"Fold {fold} | Train Size: {len(X_train)} | Test Size: {len(X_test)} | Trades Taken: {trades_taken} | Fold Win Rate: {win_rate:.1f}%")
        fold += 1

    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    
    print("\n--- Final Strategy Performance ---")
    print(f"Total Future Trades Taken: {total_trades}")
    print(f"True Out-of-Sample Win Rate: {overall_win_rate:.1f}%")

if __name__ == "__main__":
    train_walk_forward()