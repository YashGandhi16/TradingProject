import os
import pandas as pd
import yfinance as yf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "trades_scaled.csv")

def calculate_tr(df):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1)

def backfill_csv():
    if not os.path.exists(CSV_PATH):
        print(f"[!] Error: {CSV_PATH} not found.")
        return

    df_trades = pd.read_csv(CSV_PATH)
    
    # 1. Clean up any trailing commas from old manual entry
    df_trades = df_trades.loc[:, ~df_trades.columns.str.contains('^Unnamed')]
    
    print(f"Loaded {len(df_trades)} trades. Backfilling strict algorithmic features...")

    if 'VCP_ATR_Ratio' not in df_trades.columns:
        df_trades['VCP_ATR_Ratio'] = 1.0
    if 'SPY_Trend_Valid' not in df_trades.columns:
        df_trades['SPY_Trend_Valid'] = 1.0

    spy_data = yf.download('SPY', period="5y", interval="1d", progress=False)
    if not spy_data.empty:
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = spy_data.columns.get_level_values(0)
        spy_data.columns = spy_data.columns.str.lower()
        spy_data.index = spy_data.index.tz_localize(None) # FIX: Timezone strip
        spy_data['20ema'] = spy_data['close'].ewm(span=20, adjust=False).mean()

    for index, row in df_trades.iterrows():
        ticker = row['Ticker']
        trade_date = pd.to_datetime(row['Date'])
        
        print(f"Processing [{index+1}/{len(df_trades)}] {ticker} on {trade_date.date()}...")
        
        try:
            ticker_data = yf.download(ticker, period="max", interval="1d", progress=False)
            if ticker_data.empty: continue
            
            if isinstance(ticker_data.columns, pd.MultiIndex):
                ticker_data.columns = ticker_data.columns.get_level_values(0)
            ticker_data.columns = ticker_data.columns.str.lower()
            ticker_data.index = ticker_data.index.tz_localize(None) # FIX: Timezone strip
            
            hist = ticker_data.loc[:trade_date].copy()
            if len(hist) < 260: continue 

            hist['8ema'] = hist['close'].ewm(span=8, adjust=False).mean()
            hist['21ema'] = hist['close'].ewm(span=21, adjust=False).mean()
            hist['50ema'] = hist['close'].ewm(span=50, adjust=False).mean()
            
            hist['tr'] = calculate_tr(hist)
            hist['atr_3'] = hist['tr'].rolling(3).mean()
            hist['atr_15'] = hist['tr'].rolling(15).mean()

            today = hist.iloc[-1]
            epsilon = 0.998 # FIX: 0.2% tolerance for mathematically tight EMAs

            # 1. 1D Trend
            d_trend = int((today['8ema'] >= (today['21ema'] * epsilon)) and (today['21ema'] > today['50ema']))
            
            # 2. 1W Trend
            weekly_df = hist['close'].resample('W').last().to_frame()
            weekly_df['w_8ema'] = weekly_df['close'].ewm(span=8, adjust=False).mean()
            weekly_df['w_21ema'] = weekly_df['close'].ewm(span=21, adjust=False).mean()
            weekly_df['w_50ema'] = weekly_df['close'].ewm(span=50, adjust=False).mean()
            w_today = weekly_df.iloc[-1]
            w_trend = int((w_today['w_8ema'] >= (w_today['w_21ema'] * epsilon)) and (w_today['w_21ema'] > w_today['w_50ema']))

            # 3. Consolidation metrics
            last_20_days = hist.iloc[-21:-1]
            peak_date = last_20_days['high'].idxmax()
            yesterday_date = hist.iloc[-2].name # FIX: Stop analysis before breakout day
            
            consol_df = hist.loc[peak_date:yesterday_date].copy()
            
            if len(consol_df) > 1:
                consol_df['vol_smoothed'] = consol_df['volume'].rolling(3, min_periods=1).mean()
                smoothed_vols = consol_df['vol_smoothed'].dropna().values
                
                if len(smoothed_vols) > 1:
                    slope, _ = np.polyfit(np.arange(len(smoothed_vols)), smoothed_vols, 1)
                    vol_contract = int(slope < 0)
                else:
                    vol_contract = 0

                close_below_8 = int((consol_df['close'] < consol_df['8ema']).any())
                close_below_21 = int((consol_df['close'] < consol_df['21ema']).any())
                close_below_50 = int((consol_df['close'] < consol_df['50ema']).any())
                consol_duration = int(len(consol_df) >= 6)
            else:
                vol_contract, close_below_8, close_below_21, close_below_50, consol_duration = 0, 0, 0, 0, 0

            vcp_ratio = today['atr_3'] / today['atr_15'] if today['atr_15'] > 0 else 1.0

            spy_hist = spy_data.loc[:trade_date]
            spy_valid = 1
            if not spy_hist.empty:
                spy_today = spy_hist.iloc[-1]
                spy_valid = int(spy_today['close'] > spy_today['20ema'])

            # Cast to float to maintain structural integrity in the CSV
            df_trades.at[index, '1W_Trend_Valid'] = float(w_trend)
            df_trades.at[index, '1D_Trend_Valid'] = float(d_trend)
            df_trades.at[index, '1D_Volume_Contracting'] = float(vol_contract)
            df_trades.at[index, 'Consol_Close_Below_8EMA'] = float(close_below_8)
            df_trades.at[index, 'Consol_Close_Below_21EMA'] = float(close_below_21)
            df_trades.at[index, 'Consol_Close_Below_50EMA'] = float(close_below_50)
            df_trades.at[index, '>=6_days_consol'] = float(consol_duration)
            df_trades.at[index, 'VCP_ATR_Ratio'] = round(vcp_ratio, 3)
            df_trades.at[index, 'SPY_Trend_Valid'] = float(spy_valid)

        except Exception as e:
            print(f"  -> Error processing {ticker}: {e}")

    df_trades.to_csv(CSV_PATH, index=False)
    print(f"\n[+] Backfill complete. {CSV_PATH} has been updated with pure math features.")

if __name__ == "__main__":
    backfill_csv()