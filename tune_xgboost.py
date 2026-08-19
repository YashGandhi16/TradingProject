import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import warnings

# Suppress annoying warnings during grid search
warnings.filterwarnings('ignore')

def tune_model():
    print("="*60)
    print(" 🔬 XGBOOST INSTITUTIONAL HYPERPARAMETER TUNER 🔬 ")
    print("="*60)

    # 1. Load the exact same dynamic dataset as your training script
    processed_dir = "data/processed/"
    if not os.path.exists(processed_dir):
        print(f"[!] Error: {processed_dir} does not exist.")
        return

    prefix = "ALL_TRADES_HYBRID"
    all_folders = [
        os.path.join(processed_dir, d) for d in os.listdir(processed_dir) 
        if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith(prefix)
    ]
    
    if not all_folders:
        print("[!] Error: No processed datasets found.")
        return

    latest_folder = max(all_folders, key=os.path.getmtime)
    print(f"[*] Loaded Dataset: {os.path.basename(latest_folder)}")
    
    train = pd.read_parquet(os.path.join(latest_folder, "train.parquet"))
    
    # --- DATA SCAVENGER (Crucial for getting all 90 rows) ---
    val_files = ["val.parquet", "validation.parquet", "holdout.parquet"]
    for v_file in val_files:
        v_path = os.path.join(latest_folder, v_file)
        if os.path.exists(v_path):
            v_df = pd.read_parquet(v_path)
            train = pd.concat([train, v_df], ignore_index=True)

    train = train.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"[*] Total Training Rows Available for Tuning: {len(train)}\n")

    # 2. The 13 Pruned Core Features
    features = [
        '1W_Trend_Valid', 'SPY_Trend_Valid', 'price_to_8ema_delta', 
        'Sector_Trend_Valid', 'intraday_rvol', 'macro_breakout_gap_pct', 
        'macro_vol_contraction_ratio', 'Consol_Close_Below_21EMA', 
        '>=6_days_consol', 'Breakout_Open_Valid', 'VCP_ATR_Ratio', 
        'Intraday_Volume_Spike', '1D_Volume_Contracting'
    ]
    
    X_train = train[features]
    y_train = train['target']

    # 3. Define the Hyperparameter Grid tailored for Small Datasets
    # Total Combinations: 3 x 3 x 3 x 3 x 2 x 2 = 324 combinations
    param_grid = {
        'max_depth': [2, 3, 4],                  # How deep the tree goes (keep small to prevent memorization)
        'learning_rate': [0.01, 0.05, 0.1],      # How fast the model learns
        'n_estimators': [50, 100, 150],          # Number of trees
        'min_child_weight': [1, 3, 5],           # CRITICAL: Forces model to ignore isolated outlier trades
        'subsample': [0.7, 1.0],                 # Trains on a random % of rows to prevent overfitting
        'colsample_bytree': [0.6, 1.0]           # Trains on a random % of features so it doesn't just rely on Sector Trend
    }

    # 4. Set up the Base Model & Cross-Validation Engine
    base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    # Stratified K-Fold ensures every slice of data has a balanced number of Wins and Losses
    cv_engine = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We score based on 'average_precision' (PR AUC) which focuses heavily on the Win Rate of our Buy signals
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='average_precision',
        cv=cv_engine,
        verbose=1,
        n_jobs=-1  # Uses all CPU cores
    )

    print("[*] Initiating Grid Search Sequence...")
    print(f"[*] Testing {len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['n_estimators']) * len(param_grid['min_child_weight']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])} combinations across 5 data folds (Total Fits: 1620).")
    print("[*] This may take 30 to 60 seconds...\n")
    
    grid_search.fit(X_train, y_train)

    print("="*60)
    print(" 🏆 OPTIMAL HYPERPARAMETERS FOUND 🏆 ")
    print("="*60)
    
    best_params = grid_search.best_params_
    
    for key, value in best_params.items():
        print(f"{key:<20}: {value}")
        
    print("\n[*] ACTION REQUIRED:")
    print("Copy these values and paste them into your base_model inside train_robust.py")

if __name__ == "__main__":
    tune_model()