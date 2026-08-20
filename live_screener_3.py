import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
import re
import time
import requests

# Suppress yfinance warnings for clean terminal output
warnings.filterwarnings("ignore")

from src.features.build_features import build_macro_features, build_intraday_features
from src.features.merge_features import align_timeframes

# --- ANTI-BAN ARMOR ---
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
})

def calculate_ema(series, days):
    return series.ewm(span=days, adjust=False).mean()

def main():
    print("="*85)
    print(" 🦅 INSTITUTIONAL CONTINUATION SCREENER (V5 - ARMORED) 🦅 ")
    print("="*85)
    
    model_path = "data/models/calibrated_xgb_model.joblib"
    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}.")
        return
        
    model = joblib.load(model_path)
    
    # --- 1. MACRO SECTOR ENGINE ---
    print("[*] Calculating Macro Sector Rotation...")
    sector_etfs = ['XLK', 'XLE', 'XLF', 'XLY', 'XLP', 'XLV', 'XLI', 'XLC', 'XLU', 'XLRE', 'XLB', 'SPY']
    
    try:
        # Bulk download all sector ETFs with the spoofed session
        sector_data_raw = yf.download(sector_etfs, period='3mo', session=session, progress=False)['Close']
        
        sector_returns = {}
        for etf in sector_etfs:
            series = sector_data_raw[etf].dropna()
            if len(series) >= 20:
                ret_20d = (series.iloc[-1] - series.iloc[-20]) / series.iloc[-20]
                sector_returns[etf] = ret_20d
            else:
                sector_returns[etf] = 0.0
                
        spy_ret = sector_returns.get('SPY', 0.0)
        
        spy_series = sector_data_raw['SPY'].dropna()
        spy_50ema = calculate_ema(spy_series, 50)
        spy_trend_valid = 1.0 if spy_series.iloc[-2] > spy_50ema.iloc[-2] else 0.0

        sorted_sectors = sorted([e for e in sector_returns.keys() if e != 'SPY'], key=lambda x: sector_returns[x], reverse=True)
        top_3_sectors = sorted_sectors[:3]
        
        print(f"[*] TOP 3 TRENDING SECTORS (20-Day): {', '.join(top_3_sectors)}\n")
        
    except Exception as e:
        print(f"[!] Warning: Failed to fetch Sector data. Defaulting to SPY. Error: {e}")
        sector_returns = {}
        spy_ret = 0.0
        spy_trend_valid = 1.0
        top_3_sectors = []

    # --- 2. LOAD TICKER UNIVERSE ---
    ticker_file = "ticker_list.md"
    if not os.path.exists(ticker_file):
        print(f"[!] {ticker_file} not found. Please create it.")
        return
        
    with open(ticker_file, "r") as f:
        raw_lines = f.readlines()
        watchlist = [re.sub(r'[^A-Z]', '', line.upper()) for line in raw_lines]
        watchlist = [t for t in watchlist if t != '']

    print(f"[*] Scanning {len(watchlist)} Tickers from {ticker_file}...\n")
    
    # EXACT 17 BINARY FEATURES
    features_list = [
        'VCP_ATR_Ratio', '1D_Trend_Valid', 'Consol_Close_Below_50EMA', 
        'Breakout_Open_Valid', '1D_Volume_Contracting', 'intraday_rvol', 
        'macro_vol_contraction_ratio', 'macro_breakout_gap_pct', 
        'price_to_8ema_delta', '1W_Trend_Valid', 'intraday_vol_contraction_ratio', 
        'Intraday_Volume_Spike', 'Consol_Close_Below_8EMA', 
        'Consol_Close_Below_21EMA', '>=6_days_consol', 'SPY_Trend_Valid', 
        'Sector_Trend_Valid'
    ]

    sector_map = {
        'Technology': 'XLK', 'Energy': 'XLE', 'Financial Services': 'XLF',
        'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Healthcare': 'XLV',
        'Industrials': 'XLI', 'Communication Services': 'XLC', 'Utilities': 'XLU',
        'Real Estate': 'XLRE', 'Basic Materials': 'XLB'
    }

    stats = {'liquidity': 0, 'downtrend': 0, 'too_far_from_highs': 0, 'not_coiled_enough': 0, 'data_error': 0, 'low_prob': 0, 'passed': 0}
    results = []

    for ticker in watchlist:
        try:
            # Inject session into the daily download
            daily_raw = yf.download(ticker, period='1y', session=session, progress=False)
            if daily_raw.empty or len(daily_raw) < 50:
                stats['data_error'] += 1
                time.sleep(0.2) # Throttle on failure
                continue
                
            if isinstance(daily_raw.columns, pd.MultiIndex):
                daily_raw.columns = daily_raw.columns.droplevel(1)

            daily_raw.rename(columns=str.lower, inplace=True)
            daily_raw.index = daily_raw.index.tz_localize(None)

            current_price = daily_raw['close'].iloc[-1]
            daily_raw['50EMA'] = calculate_ema(daily_raw['close'], 50)
            daily_raw['Vol_20SMA'] = daily_raw['volume'].rolling(20).mean()
            
            adv = daily_raw['Vol_20SMA'].iloc[-1] * current_price
            if adv < 5_000_000:
                stats['liquidity'] += 1
                time.sleep(0.1)
                continue
                
            if current_price < daily_raw['50EMA'].iloc[-1]:
                stats['downtrend'] += 1
                time.sleep(0.1)
                continue
                
            last_20_days = daily_raw.iloc[-21:-1]
            peak_high = last_20_days['high'].max()
            if current_price < (peak_high * 0.92):
                stats['too_far_from_highs'] += 1
                time.sleep(0.1)
                continue

            peak_idx = last_20_days['high'].argmax()
            days_since_peak = len(last_20_days) - 1 - peak_idx
            if days_since_peak < 5:
                stats['not_coiled_enough'] += 1
                time.sleep(0.1)
                continue
            
            last_5_days = daily_raw.iloc[-6:-1]
            base_width = (last_5_days['high'].max() - last_5_days['low'].min()) / last_5_days['low'].min()
            
            # Inject session into the intraday download
            intraday_raw = yf.download(ticker, period='5d', interval='5m', session=session, progress=False)
            if intraday_raw.empty:
                stats['data_error'] += 1
                time.sleep(0.2) # Throttle on failure
                continue
                
            if isinstance(intraday_raw.columns, pd.MultiIndex):
                intraday_raw.columns = intraday_raw.columns.droplevel(1)
            intraday_raw.rename(columns=str.lower, inplace=True)
            intraday_raw.index = intraday_raw.index.tz_localize(None)

            daily_features = build_macro_features(daily_raw)
            intraday_features = build_intraday_features(intraday_raw)
            merged_df = align_timeframes(intraday_df=intraday_features, macro_df=daily_features)
            live_bar = merged_df.tail(1).copy()
            
            daily_raw['8EMA'] = calculate_ema(daily_raw['close'], 8)
            daily_raw['21EMA'] = calculate_ema(daily_raw['close'], 21)
            
            prior_day = daily_raw.iloc[-2]
            today = daily_raw.iloc[-1]
            
            try:
                # Add spoofing to the Ticker info call
                info_ticker = yf.Ticker(ticker, session=session)
                sector_name = info_ticker.info.get('sector', 'Unknown')
                target_etf = sector_map.get(sector_name, 'SPY')
            except:
                target_etf = 'SPY'
                
            sector_trend_valid = 1.0 if sector_returns.get(target_etf, -1) > spy_ret else 0.0
            
            # Binary Math matching your working branch
            live_bar['1W_Trend_Valid'] = 1.0 if prior_day['close'] > prior_day['50EMA'] else 0.0
            live_bar['1D_Trend_Valid'] = 1.0 if prior_day['close'] > prior_day['21EMA'] else 0.0
            live_bar['1D_Volume_Contracting'] = 1.0 if prior_day['volume'] < prior_day['Vol_20SMA'] else 0.0
            live_bar['Breakout_Open_Valid'] = 1.0 if today['open'] >= prior_day['close'] else 0.0
            live_bar['Consol_Close_Below_8EMA'] = 1.0 if prior_day['close'] > prior_day['8EMA'] else 0.0
            live_bar['Consol_Close_Below_21EMA'] = 1.0 if prior_day['close'] > prior_day['21EMA'] else 0.0
            live_bar['Consol_Close_Below_50EMA'] = 1.0 if prior_day['close'] > prior_day['50EMA'] else 0.0
            live_bar['Intraday_Volume_Spike'] = 1.0 if today['volume'] > (prior_day['Vol_20SMA'] * 1.5) else 0.0
            live_bar['>=6_days_consol'] = 1.0 if days_since_peak >= 5 else 0.0
            live_bar['SPY_Trend_Valid'] = spy_trend_valid
            live_bar['Sector_Trend_Valid'] = sector_trend_valid
            
            live_bar = live_bar.replace([np.inf, -np.inf], np.nan).fillna(0)
            for col in features_list:
                if col not in live_bar.columns:
                    live_bar[col] = 0.0
                    
            X_live = live_bar[features_list]
            prob = model.predict_proba(X_live)[:, 1][0]
            
            if prob >= 0.35:
                status = "🔥 ACTIVE BREAKOUT" if current_price >= peak_high else "⏳ PRE-BREAKOUT"
                signal = "🟢 BUY" if prob >= 0.50 else "🟡 WATCH"
                
                sector_display = f"{target_etf} (TOP 3) 🌟" if target_etf in top_3_sectors else target_etf
                
                results.append({
                    'Ticker': ticker,
                    'Signal': signal,
                    'Prob': prob,
                    'Status': status,
                    'Days Coiled': days_since_peak,
                    'Base Width': f"{base_width*100:.1f}%",
                    'Sector': sector_display
                })
                stats['passed'] += 1
            else:
                stats['low_prob'] += 1
                
            # Throttle the loop on successful scans
            time.sleep(0.2)
            
        except Exception as e:
            # Unmasked the error so you can see if something else breaks
            # print(f"[!] Crash on {ticker}: {type(e).__name__} - {str(e)}") 
            stats['data_error'] += 1
            time.sleep(0.2)

    # --- PRINT DASHBOARD ---
    print(f"{'TICKER':<8} | {'SIGNAL':<8} | {'PROB':<5} | {'STATUS':<18} | {'DAYS COILED':<11} | {'BASE WIDTH':<10} | {'SECTOR'}")
    print("-" * 85)
    
    results = sorted(results, key=lambda x: x['Prob'], reverse=True)
    
    for r in results:
        prob_str = f"{r['Prob']*100:.1f}%"
        print(f"{r['Ticker']:<8} | {r['Signal']:<8} | {prob_str:<5} | {r['Status']:<18} | {r['Days Coiled']:<11} | {r['Base Width']:<10} | {r['Sector']}")

    # --- PRINT GRAVEYARD TELEMETRY ---
    print("\n" + "="*85)
    print(" 🪦 SCREENER GRAVEYARD REPORT 🪦 ")
    print(f" Total Tickers Analyzed: {len(watchlist)}")
    print(f" - Rejected: {stats['liquidity']} (Illiquid / ADV < $5M)")
    print(f" - Rejected: {stats['downtrend']} (Downtrend / Below 50 EMA)")
    print(f" - Rejected: {stats['too_far_from_highs']} (Dead Chart / >8% away from highs)")
    print(f" - Rejected: {stats['not_coiled_enough']} (Not Coiled Enough / < 5 Days)")
    print(f" - Rejected: {stats['low_prob']} (Ugly Math / XGBoost < 35% Confidence)")
    print(f" - API Errors: {stats['data_error']}")
    print(f" ✅ Setups Passed to Dashboard: {stats['passed']}")
    print("="*85)

if __name__ == "__main__":
    main()