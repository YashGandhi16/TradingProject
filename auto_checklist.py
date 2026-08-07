import os
import pandas as pd
import numpy as np
import yfinance as yf

def calculate_ema(series, days):
    return series.ewm(span=days, adjust=False).mean()

def main():
    print("="*60)
    print(" 🤖 AUTO-GRADER: TRUE SECTOR RS (20-DAY) & CHECKLIST 🤖 ")
    print("="*60)

    csv_path = "data/trades_scaled.csv"
    if not os.path.exists(csv_path):
        print(f"[!] Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    checklist_cols = [
        '1W_Trend_Valid', '1D_Trend_Valid', '1D_Volume_Contracting',
        'Breakout_Open_Valid', 'Intraday_Volume_Spike',
        'Consol_Close_Below_8EMA', 'Consol_Close_Below_21EMA',
        'Consol_Close_Below_50EMA', '>=6_days_consol', 'SPY_Trend_Valid',
        'Sector_Trend_Valid' 
    ]

    for col in checklist_cols:
        if col not in df.columns:
            df[col] = np.nan

    # --- SECTOR TO ETF MAPPING ---
    sector_to_etf = {
        'Technology': 'XLK', 'Energy': 'XLE', 'Financial Services': 'XLF',
        'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Healthcare': 'XLV',
        'Industrials': 'XLI', 'Communication Services': 'XLC', 'Utilities': 'XLU',
        'Real Estate': 'XLRE', 'Basic Materials': 'XLB'
    }

    etfs_to_fetch = ['SPY', 'XLK', 'XLE', 'XLF', 'XLY', 'XLP', 'XLV', 'XLI', 'XLC', 'XLU', 'XLRE', 'XLB']
    etf_data = {}
    
    print("[*] Fetching Market & Sector ETF historical data...")
    for etf in etfs_to_fetch:
        try:
            temp_df = yf.download(etf, period='5y', progress=False)
            if isinstance(temp_df.columns, pd.MultiIndex):
                close_series = temp_df['Close'][etf]
            else:
                close_series = temp_df['Close']
            
            clean_df = pd.DataFrame({'close': close_series})
            clean_df['50EMA'] = calculate_ema(clean_df['close'], 50)
            etf_data[etf] = clean_df
            print(f"  [+] {etf} loaded.")
        except Exception as e:
            print(f"  [!] Failed to load {etf}: {e}")
            etf_data[etf] = None
    print()

    count = 0
    ticker_cache = {} 

    for index, row in df.iterrows():
        # Force grade every row to update the old 1.0s
        needs_grading = True

        if needs_grading:
            trade_id = row['Trade_ID'] if 'Trade_ID' in row and pd.notna(row['Trade_ID']) else f"{row['Ticker']}_{index}"
            ticker = str(row['Ticker']).upper().strip()
            daily_path = f"data/raw_dataset/{trade_id}/daily.csv"
            
            if not os.path.exists(daily_path):
                print(f"  [!] Skipping {trade_id}: No daily.csv found.")
                continue

            try:
                daily = pd.read_csv(daily_path)
                if len(daily) < 50:
                    continue

                daily['8EMA'] = calculate_ema(daily['close'], 8)
                daily['21EMA'] = calculate_ema(daily['close'], 21)
                daily['50EMA'] = calculate_ema(daily['close'], 50)
                daily['Vol_20SMA'] = daily['volume'].rolling(20).mean()

                prior_day = daily.iloc[-2]
                breakout_day = daily.iloc[-1]
                
                df.at[index, '1W_Trend_Valid'] = 1.0 if prior_day['close'] > prior_day['50EMA'] else 0.0
                df.at[index, '1D_Trend_Valid'] = 1.0 if prior_day['close'] > prior_day['21EMA'] else 0.0
                df.at[index, '1D_Volume_Contracting'] = 1.0 if prior_day['volume'] < prior_day['Vol_20SMA'] else 0.0
                df.at[index, 'Breakout_Open_Valid'] = 1.0 if breakout_day['open'] >= prior_day['close'] else 0.0
                df.at[index, 'Consol_Close_Below_8EMA'] = 1.0 if prior_day['close'] > prior_day['8EMA'] else 0.0
                df.at[index, 'Consol_Close_Below_21EMA'] = 1.0 if prior_day['close'] > prior_day['21EMA'] else 0.0
                df.at[index, 'Consol_Close_Below_50EMA'] = 1.0 if prior_day['close'] > prior_day['50EMA'] else 0.0
                df.at[index, 'Intraday_Volume_Spike'] = 1.0 if breakout_day['volume'] > (prior_day['Vol_20SMA'] * 1.5) else 0.0

                last_6_days = daily.iloc[-7:-1]
                consol_range = (last_6_days['high'].max() - last_6_days['low'].min()) / last_6_days['low'].min()
                df.at[index, '>=6_days_consol'] = 1.0 if consol_range <= 0.06 else 0.0

                if ticker not in ticker_cache:
                    try:
                        info = yf.Ticker(ticker).info
                        sector_name = info.get('sector', 'Unknown')
                        assigned_etf = sector_to_etf.get(sector_name, 'SPY')
                        ticker_cache[ticker] = assigned_etf
                    except Exception:
                        ticker_cache[ticker] = 'SPY'
                
                target_etf = ticker_cache[ticker]

                entry_date_str = str(row['Date']) if 'Date' in row and pd.notna(row['Date']) else str(row['Entry_Time']).split(' ')[0]
                entry_date = pd.to_datetime(entry_date_str).tz_localize(None)

                spy_df = etf_data.get('SPY')
                sector_df = etf_data.get(target_etf)
                
                # SPY 50-EMA Trend Check
                if spy_df is not None and not spy_df.empty:
                    spy_history = spy_df[spy_df.index <= entry_date]
                    if not spy_history.empty:
                        spy_prior = spy_history.iloc[-1]
                        df.at[index, 'SPY_Trend_Valid'] = 1.0 if spy_prior['close'] > spy_prior['50EMA'] else 0.0
                    else:
                        df.at[index, 'SPY_Trend_Valid'] = 1.0
                else:
                    df.at[index, 'SPY_Trend_Valid'] = 1.0

                # --- THE NEW MATH: 20-DAY OUTPERFORMANCE ---
                if spy_df is not None and sector_df is not None and not spy_history.empty:
                    sector_history = sector_df[sector_df.index <= entry_date]
                    
                    if len(spy_history) >= 20 and len(sector_history) >= 20:
                        spy_20d_ret = (spy_history['close'].iloc[-1] - spy_history['close'].iloc[-20]) / spy_history['close'].iloc[-20]
                        sector_20d_ret = (sector_history['close'].iloc[-1] - sector_history['close'].iloc[-20]) / sector_history['close'].iloc[-20]
                        
                        # 1.0 ONLY if the sector outpaced the broader market over the last month
                        df.at[index, 'Sector_Trend_Valid'] = 1.0 if sector_20d_ret > spy_20d_ret else 0.0
                    else:
                        df.at[index, 'Sector_Trend_Valid'] = 0.0
                else:
                    df.at[index, 'Sector_Trend_Valid'] = 0.0

                count += 1
                print(f"[+] Auto-graded: {trade_id} (Sector ETF: {target_etf})")

            except Exception as e:
                print(f"  [!] Error processing {trade_id}: {e}")

    df.to_csv(csv_path, index=False)
    print("="*60)
    print(f"[+] Complete. Successfully added True Sector RS to {count} historical trades.")

if __name__ == "__main__":
    main()