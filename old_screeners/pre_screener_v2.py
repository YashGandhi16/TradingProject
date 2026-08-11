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
        return []
    return list(set(tickers))

def calculate_tr(df):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1)

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

    print(f"Fetching broader market context (SPY)...")
    spy_data = yf.download('SPY', period="3mo", interval="1d", progress=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    spy_data.columns = spy_data.columns.str.lower()
    spy_data['20ema'] = spy_data['close'].ewm(span=20, adjust=False).mean()
    spy_valid = int(spy_data['close'].iloc[-1] > spy_data['20ema'].iloc[-1])

    if spy_valid == 0:
        print("[-] Market Warning: SPY is below its 20 EMA. Breakout failure rates are higher.")
    else:
        print("[+] Market Context: SPY is above 20 EMA. Favorable breakout environment.")

    print(f"\nScanning {len(tickers)} tickers for PRE-BREAKOUT setups...")
    data = yf.download(tickers, period="2y", interval="1d", group_by="ticker", progress=False)

    pre_breakout_candidates = []

    for ticker in tickers:
        try:
            df = data[ticker].copy() if len(tickers) > 1 else data.copy()
            if df.empty or len(df) < 260:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()
            df = df.dropna()

            # --- 1D Indicators ---
            df['8ema'] = df['close'].ewm(span=8, adjust=False).mean()
            df['21ema'] = df['close'].ewm(span=21, adjust=False).mean()
            df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
            df['rolling_high_20'] = df['high'].rolling(20).max().shift(1)
            df['vol_20ma'] = df['volume'].rolling(20).mean()
            
            # ATR Calculation
            df['tr'] = calculate_tr(df)
            df['atr_3'] = df['tr'].rolling(3).mean()
            df['atr_15'] = df['tr'].rolling(15).mean()

            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            
            # --- Pre-Breakout Filters ---
            dist_to_breakout = (today['rolling_high_20'] - today['close']) / today['rolling_high_20']
            is_near_breakout = 0.0 <= dist_to_breakout <= proximity_threshold
            holding_ema = (today['close'] >= today['8ema']) and (today['8ema'] >= today['21ema'])

            if is_near_breakout and holding_ema:
                
                # --- MATHEMATICAL CHECKLIST ---
                weekly_df = df['close'].resample('W').last().to_frame()
                weekly_df['w_8ema'] = weekly_df['close'].ewm(span=8, adjust=False).mean()
                weekly_df['w_21ema'] = weekly_df['close'].ewm(span=21, adjust=False).mean()
                weekly_df['w_50ema'] = weekly_df['close'].ewm(span=50, adjust=False).mean()
                w_today = weekly_df.iloc[-1]
                w_trend_valid = int(w_today['w_8ema'] > w_today['w_21ema'] > w_today['w_50ema'])

                d_trend_valid = int(today['8ema'] > today['21ema'] > today['50ema'])

                last_20_days = df.iloc[-21:-1] 
                peak_date = last_20_days['high'].idxmax()
                consol_df = df.loc[peak_date:].copy()
                
                # Smoothed Volume Contraction
                consol_df['vol_smoothed'] = consol_df['volume'].rolling(3, min_periods=1).mean()
                smoothed_vols = consol_df['vol_smoothed'].values
                vol_contracting_consol = 0
                if len(smoothed_vols) > 1:
                    slope, _ = np.polyfit(np.arange(len(smoothed_vols)), smoothed_vols, 1)
                    vol_contracting_consol = int(slope < 0)

                # Future Variables (Hardcoded for Potential)
                breakout_open_valid = 1
                intraday_volume_spike = 1

                close_below_8 = int((consol_df['close'] < consol_df['8ema']).any())
                close_below_21 = int((consol_df['close'] < consol_df['21ema']).any())
                close_below_50 = int((consol_df['close'] < consol_df['50ema']).any())
                consol_duration_valid = int(len(consol_df) >= 6)
                
                # VCP ATR Ratio
                vcp_ratio = today['atr_3'] / today['atr_15'] if today['atr_15'] > 0 else 1.0

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
                    '>=6_days_consol': consol_duration_valid,
                    'VCP_ATR_Ratio': vcp_ratio,
                    'SPY_Trend_Valid': spy_valid
                }
                
                # Align columns securely in case the model expects a specific order
                X_inference = pd.DataFrame([inference_data]).reindex(columns=feature_names, fill_value=0)
                pred_proba = model.predict_proba(X_inference)[0][1]

                # Target Intraday Volume (20MA Vol / 78 5-min periods in a day * 1.5x multiplier for breakout spike)
                required_5min_vol = (today['vol_20ma'] / 78) * 1.5

                pct_away = dist_to_breakout * 100
                pre_breakout_candidates.append({
                    'Ticker': ticker,
                    'Close': round(today['close'], 2),
                    'Dist_To_Breakout_%': round(pct_away, 2),
                    'ATR_Compression': f"{round(vcp_ratio, 2)}x",
                    'Target_5Min_Vol': f"{int(required_5min_vol):,}",
                    'Setup_Prob_%': round(pred_proba * 100, 2)
                })

        except Exception as e:
            continue

    print("\n" + "="*80)
    print(f"{'PRE-BREAKOUT RADAR RESULTS':^80}")
    print("="*80)

    if not pre_breakout_candidates:
        print("No pre-breakout setups detected today matching criteria.")
    else:
        results_df = pd.DataFrame(pre_breakout_candidates)
        results_df = results_df.sort_values(by='Setup_Prob_%', ascending=False)
        print(results_df.to_string(index=False))
        print("="*80)
        print(f"Total Watchlist Candidates: {len(results_df)}")

if __name__ == "__main__":
    scan_pre_breakout()