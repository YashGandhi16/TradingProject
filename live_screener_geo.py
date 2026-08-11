import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "data", "sector_cache.json")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "calibrated_xgb_model.joblib")

SECTORS = {
    'Technology': 'XLK', 'Energy': 'XLE', 'Financial Services': 'XLF',
    'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Healthcare': 'XLV',
    'Industrials': 'XLI', 'Communication Services': 'XLC', 'Utilities': 'XLU',
    'Real Estate': 'XLRE', 'Basic Materials': 'XLB'
}

def load_ticker_list():
    """Loads tickers from ticker_list.md safely."""
    filepath = os.path.join(BASE_DIR, "ticker_list.md")
    try:
        tickers = []
        with open(filepath, 'r') as f:
            for line in f:
                # Strip out any whitespace, bullets, commas, or quotes
                clean_line = re.sub(r'[^A-Za-z]', '', line.strip().upper())
                if clean_line:
                    tickers.append(clean_line)
                    
        if not tickers:
            raise FileNotFoundError # Trigger default creation if empty
            
        # Return unique tickers
        return list(set(tickers))
        
    except FileNotFoundError:
        print("[!] Warning: ticker_list.md not found or empty. Creating a default one.")
        default_list = ['CRWD', 'PLTR', 'OXY', 'HOOD', 'SOFI', 'AAPL', 'NVDA', 'TSLA']
        with open(filepath, "w") as f:
            for t in default_list:
                f.write(f"{t}\n")
        return default_list

def calculate_ema(series, days):
    return series.ewm(span=days, adjust=False).mean()

def calculate_tr(df):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1)

def build_sector_environment():
    """Downloads SPY and all Sectors to calculate 1W and 1M Relative Strength."""
    print("[*] Calculating Institutional Sector Flows (1W & 1M)...")
    tickers = ['SPY'] + list(SECTORS.values())
    data = yf.download(tickers, period='2mo', progress=False)['Close']
    
    sector_data = {}
    for ticker in tickers:
        df = pd.DataFrame({'close': data[ticker]}) if isinstance(data, pd.DataFrame) else pd.DataFrame({'close': data})
        df['50EMA'] = calculate_ema(df['close'], 50)
        
        # 5-Day (1 Week) and 20-Day (1 Month) Returns
        df['1W_Ret'] = df['close'].pct_change(periods=5)
        df['1M_Ret'] = df['close'].pct_change(periods=20)
        sector_data[ticker] = df

    spy_df = sector_data['SPY']
    spy_prior = spy_df.iloc[-2]
    spy_trend_valid = 1.0 if spy_prior['close'] > spy_prior['50EMA'] else 0.0
    spy_1m = spy_df['1M_Ret'].iloc[-1]
    
    # Rank Sectors
    sector_ranks = []
    for name, etf in SECTORS.items():
        etf_df = sector_data[etf]
        etf_1m = etf_df['1M_Ret'].iloc[-1]
        rs_valid = 1.0 if etf_1m > spy_1m else 0.0
        sector_ranks.append({
            'Sector': name,
            'ETF': etf,
            '1W_Ret': etf_df['1W_Ret'].iloc[-1],
            '1M_Ret': etf_1m,
            'RS_Valid': rs_valid
        })
        
    rank_df = pd.DataFrame(sector_ranks).sort_values(by='1M_Ret', ascending=False)
    print("\n--- Top 3 Trending Sectors (1-Month) ---")
    for i, row in rank_df.head(3).iterrows():
        print(f"  {row['ETF']} ({row['Sector']}): +{row['1M_Ret']*100:.2f}%")
    print("-" * 40 + "\n")
    
    return spy_trend_valid, rank_df

def get_sector_mapping(tickers):
    """Caches yfinance sector info to make daily scans blazing fast."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    missing = [t for t in tickers if t not in cache]
    if missing:
        print(f"[*] Caching sector data for {len(missing)} new tickers. This only happens once...")
        for ticker in missing:
            try:
                info = yf.Ticker(ticker).info
                cache[ticker] = SECTORS.get(info.get('sector', 'Unknown'), 'SPY')
            except:
                cache[ticker] = 'SPY'
        
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)

    return cache

def get_market_context():
    """Determines if the market is open to project intraday volume."""
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_open = market_open <= now <= market_close and now.weekday() < 5
    
    if is_open:
        minutes_passed = (now - market_open).total_seconds() / 60
        pct_day_passed = max(0.01, minutes_passed / 390.0) # 390 mins in trading day
    else:
        pct_day_passed = 1.0 # EOD
        
    return is_open, pct_day_passed

def main():
    print("="*80)
    print(f"{'⚡ INSTITUTIONAL CALIBRATED SCREENER ⚡':^80}")
    print("="*80)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Calibrated model not found at {MODEL_PATH}")
        return
        
    model = joblib.load(MODEL_PATH)
    features_list = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [
        'VCP_ATR_Ratio', '1D_Trend_Valid', 'Consol_Close_Below_50EMA', 
        'Breakout_Open_Valid', '1D_Volume_Contracting', 'intraday_rvol', 
        'macro_vol_contraction_ratio', 'macro_breakout_gap_pct', 
        'price_to_8ema_delta', '1W_Trend_Valid', 'intraday_vol_contraction_ratio', 
        'Intraday_Volume_Spike', 'Consol_Close_Below_8EMA', 
        'Consol_Close_Below_21EMA', '>=6_days_consol', 'SPY_Trend_Valid', 'Sector_Trend_Valid'
    ]

    universe = load_ticker_list()
    if not universe: return
    
    spy_valid, sector_ranks = build_sector_environment()
    sector_map = get_sector_mapping(universe)
    is_open, pct_day_passed = get_market_context()
    
    print(f"[*] Downloading daily data for {len(universe)} tickers (Batch mode)...")
    data = yf.download(universe, period='6mo', progress=False)
    
    results = []
    
    # Process each ticker
    for ticker in universe:
        try:
            # Handle MultiIndex extraction gracefully
            if isinstance(data.columns, pd.MultiIndex):
                df = pd.DataFrame({
                    'open': data['Open'][ticker], 'high': data['High'][ticker],
                    'low': data['Low'][ticker], 'close': data['Close'][ticker],
                    'volume': data['Volume'][ticker]
                }).dropna()
            else:
                df = data.copy() # Fallback for single ticker
                df.columns = df.columns.str.lower()
                
            if len(df) < 50: continue
            
            # Core Math
            df['8ema'] = calculate_ema(df['close'], 8)
            df['21ema'] = calculate_ema(df['close'], 21)
            df['50ema'] = calculate_ema(df['close'], 50)
            df['vol_20ma'] = df['volume'].rolling(20).mean()
            df['rolling_high_20'] = df['high'].rolling(20).max().shift(1)
            
            df['tr'] = calculate_tr(df)
            df['atr_3'] = df['tr'].rolling(3).mean()
            df['atr_15'] = df['tr'].rolling(15).mean()

            prior = df.iloc[-2]
            today = df.iloc[-1]
            last_6_days = df.iloc[-7:-1]
            
            last_21_days = df.iloc[-22:-1] # Look back over the past month (excluding today)
            peak_high = last_21_days['high'].max()
            
            # Find exactly how many days ago the high was set
            # argmax() gives the index position. If it was yesterday, days_since_peak = 0.
            peak_idx = last_21_days['high'].argmax()
            days_since_peak = len(last_21_days) - 1 - peak_idx 
            
            # FILTER 1: Must consolidate for at least 5 days (Kills 1-2 day pauses)
            if days_since_peak < 5:
                continue 
                
            # FILTER 2: The base must be tightening (Volatility Contraction)
            # Measures the absolute high-to-low range of the last 5 days
            last_5_days = df.iloc[-6:-1]
            consol_tightness = (last_5_days['high'].max() - last_5_days['low'].min()) / last_5_days['low'].min()
            
            # If the 5-day range is wider than 8%, it's too loose/choppy to be a true wedge or pennant
            if consol_tightness > 0.08: 
                continue
            
            # Setup State Definitions (The Pivot)
            dist_to_high = (today['close'] - peak_high) / peak_high
            
            if dist_to_high > 0:
                state = "🔥 ACTIVE BREAKOUT"
            elif -0.03 <= dist_to_high <= 0:
                state = "⏳ PRE-BREAKOUT"
            else:
                continue # Skip stocks nowhere near highs to save processing
                
            # Volume Projection
            projected_vol = today['volume'] / pct_day_passed
            rvol = projected_vol / today['vol_20ma'] if today['vol_20ma'] > 0 else 1.0
            vol_spike = 1.0 if projected_vol > (prior['vol_20ma'] * 1.5) else 0.0
            
            # VCP & Trend Checks
            vcp_ratio = today['atr_3'] / today['atr_15'] if today['atr_15'] > 0 else 1.0
            w_trend = 1.0 if prior['close'] > prior['50ema'] else 0.0
            d_trend = 1.0 if prior['close'] > prior['21ema'] else 0.0
            vol_contract = 1.0 if prior['volume'] < prior['vol_20ma'] else 0.0
            breakout_open = 1.0 if today['open'] >= prior['close'] else 0.0
            cb_8 = 1.0 if prior['close'] > prior['8ema'] else 0.0
            cb_21 = 1.0 if prior['close'] > prior['21ema'] else 0.0
            cb_50 = 1.0 if prior['close'] > prior['50ema'] else 0.0
            
            consol_range = (last_6_days['high'].max() - last_6_days['low'].min()) / last_6_days['low'].min()
            consol_valid = 1.0 if consol_range <= 0.06 else 0.0
            
            # Map Sector RS
            target_etf = sector_map.get(ticker, 'SPY')
            etf_rs_val = sector_ranks.loc[sector_ranks['ETF'] == target_etf, 'RS_Valid'].values
            sector_rs = etf_rs_val[0] if len(etf_rs_val) > 0 else 0.0
            
            # Assemble Feature Vector
            inference_data = {
                'VCP_ATR_Ratio': vcp_ratio, '1D_Trend_Valid': d_trend, 
                'Consol_Close_Below_50EMA': cb_50, 'Breakout_Open_Valid': breakout_open, 
                '1D_Volume_Contracting': vol_contract, 'intraday_rvol': rvol, 
                'macro_vol_contraction_ratio': df['volume'].iloc[-5:].std() / df['volume'].iloc[-20:].std(),
                'macro_breakout_gap_pct': (today['open'] - prior['close']) / prior['close'],
                'price_to_8ema_delta': (today['close'] - today['8ema']) / today['8ema'],
                '1W_Trend_Valid': w_trend, 'intraday_vol_contraction_ratio': 1.0, 
                'Intraday_Volume_Spike': vol_spike, 'Consol_Close_Below_8EMA': cb_8, 
                'Consol_Close_Below_21EMA': cb_21, '>=6_days_consol': consol_valid, 
                'SPY_Trend_Valid': spy_valid, 'Sector_Trend_Valid': sector_rs
            }
            
            X_live = pd.DataFrame([inference_data]).reindex(columns=features_list, fill_value=0.0)
            prob = model.predict_proba(X_live)[:, 1][0]
            
            # Only store results meeting the >40% standard
            if prob >= 0.40:
                signal = "🟢 BUY" if prob >= 0.50 else "🟡 WATCH"
                
                results.append({
                    'Ticker': ticker,
                    'Action': signal,
                    'Setup Phase': state,
                    'Days Coiled': f"{days_since_peak}d",
                    'Base Width': f"{(consol_tightness * 100):.1f}%",
                    'Confidence': prob,
                    'ETF': target_etf,
                    'RS': 'Strong' if sector_rs == 1.0 else 'Weak',
                    'RVOL': f"{rvol:.2f}x",
                    '% to Breakout': f"{(dist_to_high * 100):.2f}%"
                })
        except Exception:
            continue

    print("="*80)
    print(f"{'RADAR DASHBOARD (Confidence > 40%)':^80}")
    print("="*80)
    
    if not results:
        print("\nNo setups found meeting the 40% institutional threshold today.")
        print("Market may be chopping or extending. Protect capital.\n")
    else:
        results_df = pd.DataFrame(results).sort_values(by=['Confidence', 'Setup Phase'], ascending=[False, True])
        # Format confidence for display after sorting
        results_df['Confidence'] = results_df['Confidence'].apply(lambda x: f"{x*100:.1f}%")
        
        # Adjust display settings for perfect CLI alignment
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        print(results_df.to_string(index=False))
        
    print("\n" + "="*80)

if __name__ == "__main__":
    main()