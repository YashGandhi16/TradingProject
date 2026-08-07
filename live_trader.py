import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings

# Suppress yfinance warnings for clean terminal output
warnings.filterwarnings("ignore")

from src.features.build_features import build_macro_features, build_intraday_features
from src.features.merge_features import align_timeframes

def calculate_ema(series, days):
    return series.ewm(span=days, adjust=False).mean()

def fetch_sector_rs(ticker, current_date=None):
    """Dynamically calculates the 20-Day True Sector Relative Strength."""
    sector_to_etf = {
        'Technology': 'XLK', 'Energy': 'XLE', 'Financial Services': 'XLF',
        'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Healthcare': 'XLV',
        'Industrials': 'XLI', 'Communication Services': 'XLC', 'Utilities': 'XLU',
        'Real Estate': 'XLRE', 'Basic Materials': 'XLB'
    }
    
    try:
        info = yf.Ticker(ticker).info
        sector_name = info.get('sector', 'Unknown')
        target_etf = sector_to_etf.get(sector_name, 'SPY')
    except:
        target_etf = 'SPY'

    try:
        spy_df = yf.download('SPY', period='3mo', progress=False)['Close']
        sector_df = yf.download(target_etf, period='3mo', progress=False)['Close']
        
        # Format if yfinance returns multi-index
        if isinstance(spy_df, pd.DataFrame): spy_df = spy_df['SPY']
        if isinstance(sector_df, pd.DataFrame): sector_df = sector_df[target_etf]
            
        spy_df = pd.DataFrame({'close': spy_df})
        sector_df = pd.DataFrame({'close': sector_df})
        
        spy_df['50EMA'] = calculate_ema(spy_df['close'], 50)
        
        spy_prior = spy_df.iloc[-2] # Look at yesterday for trend validation
        spy_trend_valid = 1.0 if spy_prior['close'] > spy_prior['50EMA'] else 0.0
        
        spy_20d_ret = (spy_df['close'].iloc[-1] - spy_df['close'].iloc[-20]) / spy_df['close'].iloc[-20]
        sector_20d_ret = (sector_df['close'].iloc[-1] - sector_df['close'].iloc[-20]) / sector_df['close'].iloc[-20]
        
        sector_trend_valid = 1.0 if sector_20d_ret > spy_20d_ret else 0.0
        
        return spy_trend_valid, sector_trend_valid, target_etf
    except Exception as e:
        return 1.0, 0.0, 'SPY' # Safe defaults if API fails

def main():
    print("="*65)
    print(" ⚡ LIVE QUANTITATIVE EXECUTION ENGINE ⚡ ")
    print("="*65)
    
    # 1. Load Calibrated Model
    model_path = "data/models/calibrated_xgb_model.joblib"
    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}. Train the model first.")
        return
        
    model = joblib.load(model_path)
    print(f"[*] Calibrated Model Loaded Successfully.")
    
    # 2. Define your daily watchlist here
    watchlist = ['CRWD', 'PLTR', 'OXY', 'HOOD', 'SOFI']
    print(f"[*] Scanning Watchlist: {', '.join(watchlist)}\n")
    
    features_list = [
        'VCP_ATR_Ratio', '1D_Trend_Valid', 'Consol_Close_Below_50EMA', 
        'Breakout_Open_Valid', '1D_Volume_Contracting', 'intraday_rvol', 
        'macro_vol_contraction_ratio', 'macro_breakout_gap_pct', 
        'price_to_8ema_delta', '1W_Trend_Valid', 'intraday_vol_contraction_ratio', 
        'Intraday_Volume_Spike', 'Consol_Close_Below_8EMA', 
        'Consol_Close_Below_21EMA', '>=6_days_consol', 'SPY_Trend_Valid', 
        'Sector_Trend_Valid'
    ]

    for ticker in watchlist:
        print(f"--- Analyzing {ticker} ---")
        try:
            # Fetch Data
            daily_raw = yf.download(ticker, period='1y', progress=False)
            intraday_raw = yf.download(ticker, period='5d', interval='5m', progress=False)
            
            if daily_raw.empty or intraday_raw.empty:
                print(f"  [!] Failed to fetch data for {ticker}. Skipping.")
                continue
                
            # Clean yfinance multi-index columns if present
            if isinstance(daily_raw.columns, pd.MultiIndex):
                daily_raw.columns = daily_raw.columns.droplevel(1)
                intraday_raw.columns = intraday_raw.columns.droplevel(1)

            # Rename columns to match pipeline lowercase standard
            daily_raw.rename(columns=str.lower, inplace=True)
            intraday_raw.rename(columns=str.lower, inplace=True)
            daily_raw.index = daily_raw.index.tz_localize(None)
            intraday_raw.index = intraday_raw.index.tz_localize(None)

            # Build core features using your existing pipeline
            daily_features = build_macro_features(daily_raw)
            intraday_features = build_intraday_features(intraday_raw)
            merged_df = align_timeframes(intraday_df=intraday_features, macro_df=daily_features)
            
            # Isolate the current live bar
            live_bar = merged_df.tail(1).copy()
            
            # Generate Checklist Features
            daily_raw['8EMA'] = calculate_ema(daily_raw['close'], 8)
            daily_raw['21EMA'] = calculate_ema(daily_raw['close'], 21)
            daily_raw['50EMA'] = calculate_ema(daily_raw['close'], 50)
            daily_raw['Vol_20SMA'] = daily_raw['volume'].rolling(20).mean()
            
            prior_day = daily_raw.iloc[-2]
            today = daily_raw.iloc[-1]
            last_6_days = daily_raw.iloc[-7:-1]
            
            live_bar['1W_Trend_Valid'] = 1.0 if prior_day['close'] > prior_day['50EMA'] else 0.0
            live_bar['1D_Trend_Valid'] = 1.0 if prior_day['close'] > prior_day['21EMA'] else 0.0
            live_bar['1D_Volume_Contracting'] = 1.0 if prior_day['volume'] < prior_day['Vol_20SMA'] else 0.0
            live_bar['Breakout_Open_Valid'] = 1.0 if today['open'] >= prior_day['close'] else 0.0
            live_bar['Consol_Close_Below_8EMA'] = 1.0 if prior_day['close'] > prior_day['8EMA'] else 0.0
            live_bar['Consol_Close_Below_21EMA'] = 1.0 if prior_day['close'] > prior_day['21EMA'] else 0.0
            live_bar['Consol_Close_Below_50EMA'] = 1.0 if prior_day['close'] > prior_day['50EMA'] else 0.0
            live_bar['Intraday_Volume_Spike'] = 1.0 if today['volume'] > (prior_day['Vol_20SMA'] * 1.5) else 0.0
            
            consol_range = (last_6_days['high'].max() - last_6_days['low'].min()) / last_6_days['low'].min()
            live_bar['>=6_days_consol'] = 1.0 if consol_range <= 0.06 else 0.0
            
            # Get True Sector RS
            spy_valid, sector_valid, etf = fetch_sector_rs(ticker)
            live_bar['SPY_Trend_Valid'] = spy_valid
            live_bar['Sector_Trend_Valid'] = sector_valid
            
            # Clean and align features for the model
            live_bar = live_bar.replace([np.inf, -np.inf], np.nan).fillna(0)
            for col in features_list:
                if col not in live_bar.columns:
                    live_bar[col] = 0.0
                    
            X_live = live_bar[features_list]
            
            # Predict
            prob = model.predict_proba(X_live)[:, 1][0]
            
            if prob >= 0.50:
                print(f"  🟢 SIGNAL: BUY  | Confidence: {prob*100:.1f}% | Sector: {etf} (RS: {int(sector_valid)})")
            else:
                print(f"  🔴 SIGNAL: PASS | Confidence: {prob*100:.1f}% | Sector: {etf} (RS: {int(sector_valid)})")
                
        except Exception as e:
            print(f"  [!] Execution Error on {ticker}: {e}")
            
    print("\n" + "="*65)
    print("Scan Complete.")

if __name__ == "__main__":
    main()