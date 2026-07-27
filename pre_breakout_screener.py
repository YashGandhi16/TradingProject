import os
import re
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_PATH = os.path.join(BASE_DIR, "ticker_list.md")

def parse_watchlist(filepath):
    tickers = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                clean_line = re.sub(r'[^A-Za-z]', '', line.strip().upper())
                if clean_line:
                    tickers.append(clean_line)
    except FileNotFoundError:
        print(f"[!] Error: {filepath} not found.")
        return []
    return list(set(tickers))

def scan_pre_breakout(proximity_threshold=0.03):
    """
    Scans tickers for setup candidates that are within `proximity_threshold` (default 3%)
    of their 20-day rolling high with volume contraction and solid EMA support.
    """
    tickers = parse_watchlist(WATCHLIST_PATH)
    if not tickers:
        print("No tickers found to scan.")
        return

    print(f"Scanning {len(tickers)} tickers for PRE-BREAKOUT setups (within {proximity_threshold*100:.1f}% of 20D High)...")
    data = yf.download(tickers, period="3mo", interval="1d", group_by="ticker", progress=False)

    pre_breakout_candidates = []

    for ticker in tickers:
        try:
            df = data[ticker].copy() if len(tickers) > 1 else data.copy()
            if df.empty:
                continue

            df.columns = df.columns.str.lower()
            df = df.dropna()

            if len(df) < 20:
                continue

            # Indicators
            df['8ema'] = df['close'].ewm(span=8, adjust=False).mean()
            df['21ema'] = df['close'].ewm(span=21, adjust=False).mean()
            df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
            df['rolling_high_20'] = df['high'].rolling(20).max().shift(1)
            df['vol_20ma'] = df['volume'].rolling(20).mean()

            today = df.iloc[-1]
            
            # 1. Proximity Check: Price is within 0.0% to 3.0% below the 20-day high
            dist_to_breakout = (today['rolling_high_20'] - today['close']) / today['rolling_high_20']
            is_near_breakout = 0.0 <= dist_to_breakout <= proximity_threshold

            # 2. Structure Check: Holding above 8 EMA & 21 EMA
            holding_ema = (today['close'] >= today['8ema']) and (today['8ema'] >= today['21ema'])

            # 3. Volume Contraction: Volume below 20-day average volume
            volume_contracting = today['volume'] < today['vol_20ma']

            if is_near_breakout and holding_ema and volume_contracting:
                pct_away = dist_to_breakout * 100
                pre_breakout_candidates.append({
                    'Ticker': ticker,
                    'Close': round(today['close'], 2),
                    '20D_High': round(today['rolling_high_20'], 2),
                    'Dist_To_Breakout_%': round(pct_away, 2),
                    'Vol_Vs_20MA_Ratio': round(today['volume'] / today['vol_20ma'], 2)
                })

        except Exception as e:
            continue

    # Display Results
    print("\n" + "="*60)
    print(" PRE-BREAKOUT RADAR RESULTS ")
    print("="*60)

    if not pre_breakout_candidates:
        print("No pre-breakout setups detected today matching criteria.")
    else:
        results_df = pd.DataFrame(pre_breakout_candidates)
        results_df = results_df.sort_values(by='Dist_To_Breakout_%', ascending=True)
        print(results_df.to_string(index=False))
        print("="*60)
        print(f"Total Watchlist Candidates: {len(results_df)}")

if __name__ == "__main__":
    scan_pre_breakout()