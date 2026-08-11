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

def scan_live_breakouts():
    tickers = parse_watchlist(WATCHLIST_PATH)
    if not tickers:
        print("No tickers found to scan.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model not found at {MODEL_PATH}")
        return

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    feature_names = model.get_booster().feature_names

    print(f"Scanning {len(tickers)} tickers for LIVE BREAKOUTS today...")
    
    # Download SPY for market filter
    spy_data = yf.download('SPY', period="1mo", interval="1d", progress=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    spy_data.columns = spy_data.columns.str.lower()
    spy_data.index = spy_data.index.tz_localize(None)
    spy_data['20ema'] = spy_data['close'].ewm(span=20, adjust=False).mean()
    spy_valid = int(spy_data['close'].iloc[-1] > spy_data['20ema'].iloc[-1]) if not spy_data.empty else 1

    data = yf.download(tickers, period="2y", interval="1d", group_by="ticker", progress=False)

    live_candidates = []

    for ticker in tickers:
        try:
            df = data[ticker].copy() if len(tickers) > 1 else data.copy()
            if df.empty or len(df) < 260:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()
            df.index = df.index.tz_localize(None)

            df['8ema'] = df['close'].ewm(span=8, adjust=False).mean()
            df['21ema'] = df['close'].ewm(span=21, adjust=False).mean()
            df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
            df['rolling_high_20'] = df['high'].rolling(20).max().shift(1)
            df['vol_20ma'] = df['volume'].rolling(20).mean()

            today = df.iloc[-1]
            yesterday = df.iloc[-2]

            # LIVE BREAKOUT CONDITION (Price crossed the line)
            if today['close'] > today['rolling_high_20']:
                
                # Math features matching the backfill
                weekly_df = df['close'].resample('W').last().to_frame()
                weekly_df['w_8ema'] = weekly_df['close'].ewm(span=8, adjust=False).mean()
                weekly_df['w_21ema'] = weekly_df['close'].ewm(span=21, adjust=False).mean()
                weekly_df['w_50ema'] = weekly_df['close'].ewm(span=50, adjust=False).mean()
                w_today = weekly_df.iloc[-1]
                w_trend_valid = int(w_today['w_8ema'] > w_today['w_21ema'] > w_today['w_50ema'])

                d_trend_valid = int(today['8ema'] > today['21ema'] > today['50ema'])

                last_20_days = df.iloc[-21:-1]
                peak_date = last_20_days['high'].idxmax()
                consol_df = df.loc[peak_date:yesterday.name] 
                
                if len(consol_df) > 1:
                    consol_df = consol_df.copy()
                    consol_df['vol_smoothed'] = consol_df['volume'].rolling(3, min_periods=1).mean()
                    smoothed_vols = consol_df['vol_smoothed'].values
                    if len(smoothed_vols) > 1:
                        slope, _ = np.polyfit(np.arange(len(smoothed_vols)), smoothed_vols, 1)
                        vol_contracting_consol = int(slope < 0)
                    else:
                        vol_contracting_consol = 0
                    close_below_8 = int((consol_df['close'] < consol_df['8ema']).any())
                    close_below_21 = int((consol_df['close'] < consol_df['21ema']).any())
                    close_below_50 = int((consol_df['close'] < consol_df['50ema']).any())
                    consol_duration_valid = int(len(consol_df) >= 6)
                else:
                    vol_contracting_consol, close_below_8, close_below_21, close_below_50, consol_duration_valid = 0, 0, 0, 0, 0

                breakout_open_valid = int(today['open'] > today['8ema'])
                intraday_volume_spike = int(today['volume'] > today['vol_20ma']) 

                df['tr'] = calculate_tr(df)
                df['atr_3'] = df['tr'].rolling(3).mean()
                df['atr_15'] = df['tr'].rolling(15).mean()
                
                # Extract directly from df since 'today' was defined before these columns existed
                current_atr_3 = df['atr_3'].iloc[-1]
                current_atr_15 = df['atr_15'].iloc[-1]
                vcp_ratio = current_atr_3 / current_atr_15 if current_atr_15 > 0 else 1.0

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
                
                # Fill missing model columns with 0.0 to prevent crash
                for col in feature_names:
                    if col not in inference_data:
                        inference_data[col] = 0.0

                X_inference = pd.DataFrame([inference_data])[feature_names]
                pred_proba = model.predict_proba(X_inference)[0][1]

                live_candidates.append({
                    'Ticker': ticker,
                    'Current_Price': round(today['close'], 2),
                    '20D_High': round(today['rolling_high_20'], 2),
                    'Vol_Pace': f"{int(intraday_rvol * 100)}%",
                    'Buy_Prob_%': round(pred_proba * 100, 2)
                })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    print("\n" + "="*60)
    print(" LIVE BREAKOUT DETECTOR RESULTS ")
    print("="*60)

    if not live_candidates:
        print("No active mathematical breakouts detected right now.")
    else:
        results_df = pd.DataFrame(live_candidates)
        results_df = results_df.sort_values(by='Buy_Prob_%', ascending=False)
        print(results_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    scan_live_breakouts()