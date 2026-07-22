import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# A mix of high-volume tech, growth, and standard stocks (similar to your wins)
TICKERS_TO_SCAN = [
    "AAPL", "TSLA", "NVDA", "DELL", "BB", "CAT", "PLTR", "RKLB", 
    "AMD", "AMZN", "META", "MSFT", "NFLX", "UBER", "HOOD", "COIN",
    "CRWD", "PANW", "SNOW", "DDOG", "SMCI", "ARM", "IONQ", "U"
]

def scan_for_setups(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads daily data for a ticker and flags days that match the 
    EMA Trend Continuation macro setup.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return pd.DataFrame()
            
        # Flatten multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()

    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    df['rolling_high_10'] = df['high'].rolling(window=10).max().shift(1)

    
    # Rule 1: Uptrend Alignment (8 > 21 > 50)
    uptrend_mask = (df['ema_8'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
    
    # Rule 2: Volume Contraction during consolidation
    # Were at least 3 of the last 5 days BELOW average volume?
    low_vol_days = (df['volume'] < df['vol_sma_20']).astype(int)
    consolidation_mask = low_vol_days.rolling(window=5).sum() >= 3
    
    # Rule 3: The Breakout Trigger
    # Today's close breaks the recent 10-day high AND volume is ABOVE average
    breakout_mask = (df['close'] > df['rolling_high_10']) & (df['volume'] > df['vol_sma_20'])
    
    # Combine all rules
    valid_setups = df[uptrend_mask & consolidation_mask & breakout_mask].copy()
    
    if not valid_setups.empty:
        valid_setups['ticker'] = ticker
        valid_setups['setup_date'] = valid_setups.index.strftime('%Y-%m-%d')
        # Keep only the relevant columns for the report
        return valid_setups[['ticker', 'setup_date', 'close', 'volume', 'vol_sma_20']]
    
    return pd.DataFrame()

def main():
    print("--- Starting Historical Setup Scanner ---")
    
    # Scan the last 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    all_setups = []
    
    for ticker in TICKERS_TO_SCAN:
        print(f"Scanning {ticker}...")
        setups_df = scan_for_setups(
            ticker, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        if not setups_df.empty:
            all_setups.append(setups_df)
            
    if all_setups:
        final_df = pd.concat(all_setups)
        final_df = final_df.sort_values(by='setup_date', ascending=False)
        
        output_file = "data/potential_setups.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\n✅ Success! Found {len(final_df)} historical setups.")
        print(f"Results saved to {output_file}")
        
        print("\nTop 10 most recent setups found:")
        print(final_df.head(10).to_string(index=False))
    else:
        print("\nNo setups found matching your criteria.")

if __name__ == "__main__":
    main()