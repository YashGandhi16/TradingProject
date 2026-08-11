import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import os

def train_robust():
    print("="*60)
    print(" 🧠 INSTITUTIONAL CALIBRATED XGBOOST 🧠 ")
    print("="*60)
    
    # 1. Dynamically find the latest dataset
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
        print("[!] Error: No processed datasets found. Run pipeline first.")
        return

    latest_folder = max(all_folders, key=os.path.getmtime)
    print(f"[*] Dynamically loaded latest dataset: {os.path.basename(latest_folder)}\n")
    
    train_path = os.path.join(latest_folder, "train.parquet")
    test_path = os.path.join(latest_folder, "test.parquet")
    
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    
    # --- THE DATA SCAVENGER ---
    # Look for the hidden 16 rows and inject them into the training set
    val_files = ["val.parquet", "validation.parquet", "holdout.parquet"]
    for v_file in val_files:
        v_path = os.path.join(latest_folder, v_file)
        if os.path.exists(v_path):
            v_df = pd.read_parquet(v_path)
            train = pd.concat([train, v_df], ignore_index=True)
            print(f"[*] Scavenger Discovered {v_file}! Injected {len(v_df)} extra rows into Train Set.")

    train = train.replace([np.inf, -np.inf], np.nan).fillna(0)
    test = test.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\nLoaded Train: {len(train)} rows")
    print(f"Loaded Test:  {len(test)} rows")
    print(f"Total Model Pipeline: {len(train) + len(test)} rows\n")

    # Pruned Core Features
    features = [
        '1D_Trend_Valid', 
        'intraday_rvol', 
        'price_to_8ema_delta', 
        '1D_Volume_Contracting', 
        'Consol_Close_Below_50EMA', 
        'VCP_ATR_Ratio', 
        'macro_vol_contraction_ratio', 
        'Consol_Close_Below_8EMA', 
        'macro_breakout_gap_pct'
    ]
    
    for col in features:
        if col not in train.columns:
            train[col] = 0.0
            test[col] = 0.0

    X_train, y_train = train[features], train['target']
    X_test, y_test = test[features], test['target']

    print(f"Training on {len(features)} Core Features across {len(X_train)} train rows...\n")

    # 2. Define the Base Arrogant Model
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        eval_metric='logloss'
    )
    
    # --- FEATURE IMPORTANCE SCAN ---
    base_model.fit(X_train, y_train)
    importances = base_model.feature_importances_
    feature_rank = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_rank = feature_rank.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("--- MATHEMATICAL WEIGHT RANKING ---")
    for idx, row in feature_rank.iterrows():
        weight = row['Importance'] * 100
        if weight >= 8.0:
            status = "🟩 CORE DRIVER"
        elif weight >= 3.0:
            status = "🟨 CONTRIBUTOR"
        else:
            status = "🟥 NOISE"
        print(f"{idx+1:2d}. {row['Feature']:<30} | {weight:>5.1f}% | {status}")
    print("-----------------------------------\n")
    
    # 3. Apply Platt Scaling
    print("[*] Applying Platt Scaling (CalibratedClassifierCV)...")
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # 4. Make Calibrated Predictions
    probs = calibrated_model.predict_proba(X_test)[:, 1]
    
    threshold = 0.50
    preds = (probs >= threshold).astype(int)

    # 5. Output Institutional Metrics
    print("\n--- Model Evaluation (Test Set) ---")
    print(f"Classification Report (Threshold: {threshold}):")
    print(classification_report(y_test, preds, zero_division=0))
    
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    print("Confusion Matrix:")
    print(f"True Negatives (Avoided Traps): {tn}")
    print(f"False Positives (Bad Buys):     {fp}")
    print(f"False Negatives (Missed Wins):  {fn}")
    print(f"True Positives (Good Buys):     {tp}")
    
    total_trades_taken = tp + fp
    win_rate = (tp / total_trades_taken * 100) if total_trades_taken > 0 else 0.0
    
    print("\n--- Strategy Performance ---")
    print(f"Total Trades Taken by Model: {total_trades_taken}")
    print(f"Model Win Rate:              {win_rate:.2f}%")

    print("\n--- Calibrated Trade Audit Log (Test Set) ---")
    if 'trade_id' in test.columns:
        audit_df = test[['trade_id', 'target']].copy()
        audit_df['predicted'] = preds
        audit_df['probability'] = probs
        
        trades_taken_df = audit_df[audit_df['predicted'] == 1].copy()
        
        if trades_taken_df.empty:
            print("No trades were taken based on the current threshold.")
        else:
            trades_taken_df = trades_taken_df.sort_values(by='probability', ascending=False)
            for _, row in trades_taken_df.iterrows():
                outcome_label = "✅ WIN  (True Positive)" if row['target'] == 1 else "❌ LOSS (False Positive)"
                print(f"{outcome_label} | {row['trade_id']:<18} | Calibrated Confidence: {row['probability']*100:.1f}%")
    
    import joblib
    model_path = "data/models/calibrated_xgb_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(calibrated_model, model_path)
    print(f"\nModel strictly saved to {model_path}")

if __name__ == "__main__":
    train_robust()