import os
import pandas as pd
import yfinance as yf
from datetime import timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# 1. Load environment credentials
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# 2. Define Time Windows
MACRO_DAYS_BEFORE = 730
MACRO_DAYS_AFTER = 30

INTRADAY_DAYS_BEFORE = 14
INTRADAY_DAYS_AFTER = 5

# 3. Define robust project paths using os
# This dynamically calculates the path back up to your root trading-pattern-ai/ folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "data", "trades_scaled.csv")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_dataset")

def initialize_alpaca():
    """Validates and initializes the Alpaca historical client."""
    if not API_KEY or not SECRET_KEY:
        raise ValueError("Error: Alpaca API keys not found. Check your .env file.")
    return StockHistoricalDataClient(API_KEY, SECRET_KEY)

def format_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flattens yFinance multi-level headers and standardizes columns."""
    df = df.copy()
    
    # Flatten MultiIndex columns if yFinance exported Ticker as the second level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Standardize column names
    df.columns = df.columns.str.lower()
    df.index.name = 'date'
    
    # Keep only essential columns to match our standard schema
    expected_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in expected_cols if c in df.columns]]
    
    return df

def format_alpaca_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts timestamp from multi-index and standardizes columns."""
    df = df.copy()
    
    # Alpaca returns a MultiIndex of (symbol, timestamp). We want timestamp as the sole index.
    df = df.reset_index()
    
    # Rename and set the index
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})
    df = df.set_index('date')
    
    # Standardize column names
    df.columns = df.columns.str.lower()
    
    # Drop unwanted metadata columns to match yfinance schema
    cols_to_drop = ['symbol', 'trade_count', 'vwap']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df

def download_trade_data():
    """Reads trades.csv and downloads daily, weekly, and 5-min historical data."""
    
    # Initialize Alpaca client
    alpaca_client = initialize_alpaca()
    if not alpaca_client:
        return

    # Ensure output directories exist
    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"Loading trades from {CSV_PATH}...")
    try:
        trades = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_PATH}.")
        return

    # Using Entry_Time from the updated CSV format
    trades['Entry_Time'] = pd.to_datetime(trades['Entry_Time'])

    for index, row in trades.iterrows():
        trade_id = row['Trade_ID']
        ticker = row['Ticker']
        trade_date = row['Entry_Time']

        print(f"\n--- Processing Trade {trade_id}: {ticker} on {trade_date.strftime('%Y-%m-%d %H:%M')} ---")

        # Create dedicated directory for this trade
        trade_dir = os.path.join(RAW_DIR, str(trade_id))
        os.makedirs(trade_dir, exist_ok=True)

        # Calculate macro time window (yfinance)
        macro_start = trade_date - timedelta(days=MACRO_DAYS_BEFORE)
        macro_end = trade_date + timedelta(days=MACRO_DAYS_AFTER)

        # 1. Daily Data (Yahoo Finance)
        daily_df = yf.download(ticker, start=macro_start, end=macro_end, interval="1d", progress=False)
        if not daily_df.empty:
            daily_df = format_yfinance_df(daily_df)
            daily_path = os.path.join(trade_dir, "daily.csv")
            daily_df.to_csv(daily_path)
            print("  [+] Saved daily.csv (yfinance)")
        else:
            print(f"  [!] Warning: No daily data found for {ticker}")

        # 2. Weekly Data (Yahoo Finance)
        weekly_df = yf.download(ticker, start=macro_start, end=macro_end, interval="1wk", progress=False)
        if not weekly_df.empty:
            weekly_df = format_yfinance_df(weekly_df)
            weekly_path = os.path.join(trade_dir, "weekly.csv")
            weekly_df.to_csv(weekly_path)
            print("  [+] Saved weekly.csv (yfinance)")
        else:
            print(f"  [!] Warning: No weekly data found for {ticker}")

        # Calculate intraday time window (Alpaca)
        intraday_start = trade_date - timedelta(days=INTRADAY_DAYS_BEFORE)
        intraday_end = trade_date + timedelta(days=INTRADAY_DAYS_AFTER)

        # 3. 5-Minute Intraday Data (Alpaca)
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=intraday_start,
                end=intraday_end
            )
            bars = alpaca_client.get_stock_bars(request_params)
            intraday_df = bars.df

            if not intraday_df.empty:
                intraday_df = format_alpaca_df(intraday_df)
                intraday_path = os.path.join(trade_dir, "intraday_5m.csv")
                intraday_df.to_csv(intraday_path)
                print("  [+] Saved intraday_5m.csv (Alpaca)")
            else:
                print(f"  [!] Warning: Alpaca returned empty 5-min data for {ticker}")
                
        except Exception as e:
            print(f"  [X] Alpaca download error for {ticker}: {e}")

    print("\n==========================================")
    print("All downloads complete! Check data/raw_dataset/")
    print("==========================================")

if __name__ == "__main__":
    download_trade_data()