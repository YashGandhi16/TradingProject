import os
import pandas as pd
import yfinance as yf
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# 1. Load environment credentials
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# 2. Define Time Windows
# Macro window: 2 years back for 200-day SMAs & weekly trends
MACRO_DAYS_BEFORE = 730
MACRO_DAYS_AFTER = 30

# Intraday window: 14 calendar days back (~10 trading days) gives ample 
# warm-up for an 8-period EMA on a 5-minute chart without downloading gigabytes of unneeded data.
INTRADAY_DAYS_BEFORE = 14
INTRADAY_DAYS_AFTER = 5

def initialize_alpaca():
    """Validates and initializes the Alpaca historical client."""
    if not API_KEY or not SECRET_KEY:
        print("Error: Alpaca API keys not found. Check your .env file.")
        return None
    return StockHistoricalDataClient(API_KEY, SECRET_KEY)

def download_trade_data():
    """Reads trades.csv and downloads daily, weekly, and 5-min historical data."""
    
    # Initialize Alpaca client
    alpaca_client = initialize_alpaca()
    if not alpaca_client:
        return

    # Ensure output directories exist
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trades from data/trades.csv...")
    try:
        trades = pd.read_csv("data/trades.csv")
    except FileNotFoundError:
        print("Error: Could not find data/trades.csv. Run this script from your project root.")
        return

    trades['Date'] = pd.to_datetime(trades['Date'])

    for index, row in trades.iterrows():
        trade_id = row['Trade_ID']
        ticker = row['Ticker']
        trade_date = row['Date']

        print(f"\n--- Processing Trade {trade_id}: {ticker} on {trade_date.strftime('%Y-%m-%d')} ---")

        # Create dedicated directory for this trade
        trade_dir = raw_dir / str(trade_id)
        trade_dir.mkdir(exist_ok=True)

        # Calculate macro time window (yfinance)
        macro_start = trade_date - timedelta(days=MACRO_DAYS_BEFORE)
        macro_end = trade_date + timedelta(days=MACRO_DAYS_AFTER)

        # 1. Daily Data (Yahoo Finance)
        daily_df = yf.download(ticker, start=macro_start, end=macro_end, interval="1d", progress=False)
        if not daily_df.empty:
            daily_df.to_csv(trade_dir / "daily.csv")
            print("  [+] Saved daily.csv (yfinance)")
        else:
            print(f"  [!] Warning: No daily data found for {ticker}")

        # 2. Weekly Data (Yahoo Finance)
        weekly_df = yf.download(ticker, start=macro_start, end=macro_end, interval="1wk", progress=False)
        if not weekly_df.empty:
            weekly_df.to_csv(trade_dir / "weekly.csv")
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
                intraday_df.to_csv(trade_dir / "intraday_5m.csv")
                print("  [+] Saved intraday_5m.csv (Alpaca)")
            else:
                print(f"  [!] Warning: Alpaca returned empty 5-min data for {ticker}")
                
        except Exception as e:
            print(f"  [X] Alpaca download error for {ticker}: {e}")

    print("\n==========================================")
    print("All downloads complete! Check data/raw/")
    print("==========================================")

if __name__ == "__main__":
    download_trade_data()