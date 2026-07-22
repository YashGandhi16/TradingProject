import pandas as pd
import numpy as np

def build_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates daily context features based on the EMA Trend Continuation Strategy.
    """
    df = df.copy()
    
    # 1. Standard Price/Volume Moving Averages
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    
    # --- YASH'S ALPHA FEATURES (Translated from Markdown) ---
    
    # Alpha 1: Strict EMA Alignment (8 > 21 > 50)
    # 1 if true (perfect uptrend), 0 if false (like Loss 04, Loss 07, Loss 09)
    df['ema_alignment_flag'] = ((df['ema_8'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])).astype(int)
    
    # Alpha 2: Breakout Open Position
    # Did today's daily candle open above the 8 EMA? (Seen in Win 03, Win 11, Win 13)
    df['open_above_8ema_flag'] = (df['open'] > df['ema_8']).astype(int)
    
    # Alpha 3: Trend Strength (Gap between EMAs)
    # Measures how "far" price is above the EMAs (Seen in Win 04 KEEL, Win 05 DELL)
    df['ema_8_21_spread_pct'] = (df['ema_8'] - df['ema_21']) / df['ema_21']
    df['ema_21_50_spread_pct'] = (df['ema_21'] - df['ema_50']) / df['ema_50']
    
    # Alpha 4: Volume Contraction during Consolidation
    # How many of the last 5 days had volume BELOW the 20-day average?
    # 4 or 5 means beautiful consolidation. 0 means warning (Loss 01 RGTI).
    df['is_low_vol_day'] = (df['volume'] < df['vol_sma_20']).astype(int)
    df['days_below_avg_vol_5d'] = df['is_low_vol_day'].rolling(window=5).sum()
    
    # --------------------------------------------------------
    
    # General volatility and gap features
    df['vol_contraction_ratio'] = df['volume'] / df['vol_sma_20']
    df['pdh'] = df['high'].shift(1) # Previous Day High
    df['breakout_gap_pct'] = (df['open'] - df['pdh']) / df['pdh']
    
    # Drop intermediate columns used for math
    df.drop(columns=['is_low_vol_day'], inplace=True)
    df.dropna(inplace=True)
    
    return df

def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates 5-minute intraday features.
    """
    df = df.copy()
    
    # 5-minute EMAs
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # 5-minute Volume Average
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    
    # --- YASH'S ALPHA FEATURES ---
    
    # Alpha 5: Intraday Explosive Volume Spike
    # "Massive volume spike above the cloud on the 5 minute" (Win 06 BB, Win 10 USAR)
    # A score of 3.0 means 300% of average 5m volume.
    df['intraday_rvol'] = df['volume'] / df['vol_sma_20']
    
    # Alpha 6: Pullback to 8 EMA
    # Distance between current price and the 5-min 8 EMA
    df['price_to_8ema_delta'] = (df['close'] - df['ema_8']) / df['ema_8']
    
    # -----------------------------
    
    # Standard intraday volatility ratio
    df['intraday_vol_contraction_ratio'] = df['volume'] / df['vol_sma_20']
    
    df.dropna(inplace=True)
    return df