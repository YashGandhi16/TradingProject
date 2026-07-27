import os
import re
import pandas as pd
import yfinance as yf
import xgboost as xgb
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_PATH = os.path.join(BASE_DIR, "ticker_list.md")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "xgb_model.json")

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
    tickers = parse_watchlist(WATCHLIST_PATH)
    if not tickers:
        print("No tickers found to scan.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model file not found at {MODEL_PATH}.")
        return

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    feature_names = model.get_booster().feature_names

    # Increased period to "2y" to support 1W 50EMA calculation
    print(f"Scanning {len(tickers)} tickers for PRE-BREAKOUT setups...")
    data = yf.download(tickers, period="2y", interval="1d", group_by="ticker", progress=False)

    pre_breakout_candidates = []

    for ticker in tickers:
        try:
            df = data[ticker].copy() if len(tickers) > 1 else data.copy()
            if df.empty:
                continue

            df.columns = df.columns.str.lower()
            df = df.dropna()
            
            if len(df) < 260: # Need ~250 trading days for a 50-week EMA
                continue

            # --- 1D Indicators ---
            df['8ema'] = df['close'].ewm(span=8, adjust=False).mean()
            df['21ema'] = df['close'].ewm(span=21, adjust=False).mean()
            df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
            df['rolling_high_20'] = df['high'].rolling(20).max().shift(1)
            df['vol_20ma'] = df['volume'].rolling(20).mean()

            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            
            # --- Pre-Breakout Filters ---
            dist_to_breakout = (today['rolling_high_20'] - today['close']) / today['rolling_high_20']
            is_near_breakout = 0.0 <= dist_to_breakout <= proximity_threshold
            holding_ema = (today['close'] >= today['8ema']) and (today['8ema'] >= today['21ema'])
            volume_contracting_overall = today['volume'] < today['vol_20ma']

            if is_near_breakout and holding_ema and volume_contracting_overall:
                
                # --- AUTOMATED MATHEMATICAL CHECKLIST ---
                
                # 1. 1W Trend Valid
                weekly_df = df['close'].resample('W').last().to_frame()
                weekly_df['w_8ema'] = weekly_df['close'].ewm(span=8, adjust=False).mean()
                weekly_df['w_21ema'] = weekly_df['close'].ewm(span=21, adjust=False).mean()
                weekly_df['w_50ema'] = weekly_df['close'].ewm(span=50, adjust=False).mean()
                w_today = weekly_df.iloc[-1]
                w_trend_valid = int(w_today['w_8ema'] > w_today['w_21ema'] > w_today['w_50ema'])

                # 2. 1D Trend Valid
                d_trend_valid = int(today['8ema'] > today['21ema'] > today['50ema'])

                # Define Consolidation Window (Peak in the last 20 days up to today)
                last_20_days = df.iloc[-21:-1] 
                peak_date = last_20_days['high'].idxmax()
                consol_df = df.loc[peak_date:]
                
                # 3. 1D Volume Contracting (Negative Slope)
                consol_vols = consol_df['volume'].values
                if len(consol_vols) > 1:
                    slope, _ = np.polyfit(np.arange(len(consol_vols)), consol_vols, 1)
                    vol_contracting_consol = int(slope < 0)
                else:
                    vol_contracting_consol = 0

                # 4 & 5. Future Variables (Hardcoded to 1 for Pre-Breakout Potential)
                breakout_open_valid = 1
                intraday_volume_spike = 1

                # 6, 7, & 8. Consolidation Closes Below EMAs
                close_below_8 = int((consol_df['close'] < consol_df['8ema']).any())
                close_below_21 = int((consol_df['close'] < consol_df['21ema']).any())
                close_below_50 = int((consol_df['close'] < consol_df['50ema']).any())

                # 9. >= 6 Day Consolidation (Breakout day hasn't happened, so just measure consol_df)
                consol_duration_valid = int(len(consol_df) >= 6)

                # Base mathematical inference features
                price_to_8ema_delta = (today['close'] - today['8ema']) / today['8ema']
                macro_breakout_gap_pct = (today['open'] - yesterday['close']) / yesterday['close']
                macro_vol_contraction_ratio = df['volume'].iloc[-5:].std() / df['volume'].iloc[-20:].std()
                intraday_rvol = today['volume'] / today['vol_20ma']

                inference_data = {
                    'price_to_8ema_delta': price_to_8ema_delta,
                    'macro_breakout_gap_pct': macro_breakout_gap_pct,
                    'macro_vol_contraction_ratio': macro_vol_contraction_ratio,
                    'intraday_rvol': intraday_rvol,
                    'intraday_vol_contraction_ratio': 1.0,
                    '1W_Trend_Valid': w_trend_valid,
                    '1D_Trend_Valid': d_trend_valid,
                    '1D_Volume_Contracting': vol_contracting_consol,
                    'Breakout_Open_Valid': breakout_open_valid,
                    'Intraday_Volume_Spike': intraday_volume_spike,
                    'Consol_Close_Below_8EMA': close_below_8,
                    'Consol_Close_Below_21EMA': close_below_21,
                    'Consol_Close_Below_50EMA': close_below_50,
                    '>=6_days_consol': consol_duration_valid
                }
                
                # Run Inference
                X_inference = pd.DataFrame([inference_data])[feature_names]
                pred_proba = model.predict_proba(X_inference)[0][1]

                pct_away = dist_to_breakout * 100
                pre_breakout_candidates.append({
                    'Ticker': ticker,
                    'Close': round(today['close'], 2),
                    'Dist_To_Breakout_%': round(pct_away, 2),
                    'Setup_Prob_%': round(pred_proba * 100, 2)
                })

        except Exception as e:
            continue

    # Display Ranked Results
    print("\n" + "="*60)
    print(" PRE-BREAKOUT RADAR RESULTS ")
    print("="*60)

    if not pre_breakout_candidates:
        print("No pre-breakout setups detected today matching criteria.")
    else:
        results_df = pd.DataFrame(pre_breakout_candidates)
        # Sort by the highest model probability first
        results_df = results_df.sort_values(by='Setup_Prob_%', ascending=False)
        print(results_df.to_string(index=False))
        print("="*60)
        print(f"Total Watchlist Candidates: {len(results_df)}")

if __name__ == "__main__":
    scan_pre_breakout()