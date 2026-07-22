import os
import json
import sqlite3
import pandas as pd
from datetime import datetime

def chronological_split(df: pd.DataFrame, train_pct=0.70, val_pct=0.15):
    """Splits time-series dataframe strictly by chronology."""
    df = df.sort_index()
    n = len(df)
    
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

def init_db(db_path: str = "data/registry.db"):
    """Creates the SQLite database and registry table if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # SQLite uses TEXT to store JSON strings
    create_table_query = """
    CREATE TABLE IF NOT EXISTS dataset_registry (
        dataset_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        feature_columns TEXT,
        train_path TEXT NOT NULL,
        val_path TEXT NOT NULL,
        test_path TEXT NOT NULL,
        start_date TEXT,
        end_date TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    conn.close()

def save_and_register_dataset(
    df: pd.DataFrame, 
    ticker: str, 
    dataset_version: str, 
    db_path: str = "data/registry.db"
):
    """
    Splits the data, saves it locally to Parquet, and logs it to SQLite.
    """
    # 1. Initialize the database (safe to run multiple times)
    init_db(db_path)
    
    # 2. Split the data
    train_df, val_df, test_df = chronological_split(df)
    
    # 3. Define local paths
    base_dir = f"data/processed/{ticker}_{dataset_version}"
    os.makedirs(base_dir, exist_ok=True)
    
    train_path = f"{base_dir}/train.parquet"
    val_path = f"{base_dir}/val.parquet"
    test_path = f"{base_dir}/test.parquet"
    
    # 4. Save to local disk (compressed Parquet)
    train_df.to_parquet(train_path, engine='pyarrow', compression='snappy')
    val_df.to_parquet(val_path, engine='pyarrow', compression='snappy')
    test_df.to_parquet(test_path, engine='pyarrow', compression='snappy')
    print(f"Saved {ticker} datasets to {base_dir}/")
    
    # 5. Extract Metadata
    feature_columns = [col for col in df.columns if col != 'target']
    start_date = df.index.min().isoformat()
    end_date = df.index.max().isoformat()
    dataset_id = f"{ticker}_{dataset_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 6. Register in SQLite
    insert_query = """
        INSERT INTO dataset_registry 
        (dataset_id, ticker, feature_columns, train_path, val_path, test_path, start_date, end_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(insert_query, (
            dataset_id,
            ticker,
            json.dumps(feature_columns),
            train_path,
            val_path,
            test_path,
            start_date,
            end_date
        ))
        
        conn.commit()
        print(f"Successfully registered {dataset_id} in SQLite at {db_path}.")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    # Example execution:
    # save_and_register_dataset(
    #     df=final_merged_df,
    #     ticker="SPY",
    #     dataset_version="v1_momentum"
    # )
    pass