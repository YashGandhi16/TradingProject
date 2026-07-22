import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# A mix of high-volume tech, growth, and standard stocks (similar to your wins)
TICKERS_TO_SCAN = [
    "AAPL","AEVA","AFRM","ALAB","AMAT","AMD","AMKR","AMZN",
"APLD","APP","ARM","ARRY","ASTC","ASTS","AVGO","BA",
"BABA","BAND","BBAI","BMNR","BTC","CAT","CIFR",
"CLSK","COHR","CORZ","CRCL","CRDO","CRM","CRWD","CRWV",
"CSCO","DDOG","DELL","EOSE","FLEX","GOOGL","GS",
"HIMS","HOOD","HPE","HUT","IDWM","INFQ","INTC","IONQ",
"IREN","JNJ","JOBY","JPM","LAC","LRCX","LULU","MARA",
"MRK","MRNA","MRVL","MU","NBIS","NKE","NNE","NOW",
"NVAX","NVDA","NVTS","NXPI","OKLO","ONDS","OPEN","ORCL",
"OSCR","OSS","OUST","OXY","PATH","PFE","PLTR","PLUG",
"PONY","QBTS","QCOM","QS","QUBT","RACE","RCAT","RDW",
"RGTI","RIOT","RIVN","RKLB","RKT","RR","RUN","RXRX",
"RXT","SIMO","SMCI","SMH","SMR","SNDK","SNOW","SOFI",
"SOUN","SPCE","SPCX","SPIR","TE","TGT","TM",
"TSLA","TSM","TT","TWLO","UMAC","UNH","UPST","USAR",
"VELO","WM","WMT","WOLF","WULF","XOM","XRP","ZETA",
"ZS"
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
    df['rolling_low_10'] = df['low'].rolling(window=10).min()

    
    # Rule 1: Strict Uptrend Alignment
    # EMAs must be stacked (8 > 21 > 50) for the last 7 straight days
    daily_uptrend = (df['ema_8'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
    uptrend_mask = daily_uptrend.rolling(window=7).sum() >= 7
    
    # Rule 1.5: Shallow Pullback Limit
    # The lowest price in the last 10 days must stay above the 50 EMA
    shallow_pullback_mask = df['rolling_low_10'] > df['ema_50']
    
    # Rule 2: Volume Contraction during consolidation
    # Were at least 4 of the last 7 days BELOW average volume?
    low_vol_days = (df['volume'] < df['vol_sma_20']).astype(int)
    consolidation_mask = low_vol_days.rolling(window=7).sum() >= 4
    
    # Rule 3: The Breakout Trigger
    # Today's close breaks the recent 10-day high AND volume is ABOVE average
    breakout_mask = (df['close'] > df['rolling_high_10']) & (df['volume'] > df['vol_sma_20'])
    
    # Combine all rules
    valid_setups = df[uptrend_mask & shallow_pullback_mask & consolidation_mask & breakout_mask].copy()
    
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
        
        output_file = "data/v3_setups.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\n✅ Success! Found {len(final_df)} historical setups.")
        print(f"Results saved to {output_file}")
        
        print("\nTop 10 most recent setups found:")
        print(final_df.head(10).to_string(index=False))
    else:
        print("\nNo setups found matching your criteria.")

if __name__ == "__main__":
    main()