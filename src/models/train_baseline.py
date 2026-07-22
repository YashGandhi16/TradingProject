import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_score

def main():
    print("--- Starting XGBoost Hybrid Training ---")

    # 1. Query the Registry for the latest datasets
    print("Fetching dataset paths from registry.db...")
    try:
        conn = sqlite3.connect("data/registry.db")
        # Get the most recently registered dataset
        query = "SELECT train_path, test_path FROM dataset_registry ORDER BY created_at DESC LIMIT 1;"
        paths_df = pd.read_sql(query, conn)
        conn.close()
        
        if paths_df.empty:
            print("Error: No datasets found in registry. Run pipeline first.")
            return
            
        train_path = paths_df['train_path'].iloc[0]
        test_path = paths_df['test_path'].iloc[0]
        print(f"Loaded Train: {train_path}")
        print(f"Loaded Test: {test_path}")
        
    except Exception as e:
        print(f"Database error: {e}")
        return

    # 2. Load the Parquet files into DataFrames
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 3. Target and feature separation
    target_col = 'target'
    
    if target_col not in train_df.columns:
        print(f"Error: Target column missing. Available: {train_df.columns.tolist()}")
        return

    # Exclude metadata and absolute values. 
    # Notice we DO NOT exclude the new checklist features (e.g. 1W_Trend_Valid)
    exclude_cols = [
        target_col, 'Outcome', 'ticker', 'trade_id', 
        'open', 'high', 'low', 'close', 'volume',
        'macro_open', 'macro_high', 'macro_low', 'macro_close', 'macro_volume',
        'ema_8', 'ema_21', 'ema_50', 'vol_sma_20',
        'macro_ema_8', 'macro_ema_21', 'macro_ema_50', 'macro_vol_sma_20',
        'pdh', 'macro_pdh', 'rolling_high_20'
    ]
    
    exclude_cols = [c for c in exclude_cols if c in train_df.columns]
    features = [c for c in train_df.columns if c not in exclude_cols]

    print(f"\nTraining on {len(features)} hybrid features (Math + Checklist) across {len(train_df)} rows...")
    
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    # Calculate class imbalance
    pos_count = len(y_train[y_train == 1])
    neg_count = len(y_train[y_train == 0])
    imbalance_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # 4. Initialize and Train the XGBoost Model
    # Since we have high-signal manual features, we can keep the model relatively simple
    model = xgb.XGBClassifier(
        n_estimators=50,       
        max_depth=4,           
        learning_rate=0.05,    
        scale_pos_weight=imbalance_weight, 
        min_child_weight=1,    
        gamma=0.1,               
        random_state=42,       
        eval_metric='logloss'
    )

    print("Fitting model...")
    model.fit(X_train, y_train)

    # 5. Evaluate the Model
    print("\n--- Model Evaluation (Test Set) ---")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    print(f"Model Precision (Buy Signal Accuracy): {precision:.2%}")
    
    print("\nConfusion Matrix (Actual vs Predicted):")
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        print(f"True Negatives (Avoided Losses): {cm[0][0]}")
        print(f"False Positives (Bad Buy Signals): {cm[0][1]}")
        print(f"False Negatives (Missed Winners): {cm[1][0]}")
        print(f"True Positives (Good Buy Signals): {cm[1][1]}")
    else:
        print(cm)

    # 6. Feature Importance Extract
    print("\n--- Top 15 Most Important Features ---")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print(importance_df.head(15).to_string(index=False))
    
    # Highlight how well the manual checklist performed
    manual_features_used = [f for f in features if any(x in f for x in ['Valid', 'Contracting', 'Spike', 'Consol'])]
    print(f"\nYour Manual Checklist Features provided {importance_df[importance_df['Feature'].isin(manual_features_used)]['Importance'].sum():.1%} of the model's total decision power!")
    
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    main()