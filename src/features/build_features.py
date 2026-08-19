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
    
    # Alpha features
    df['ema_alignment_flag'] = ((df['ema_8'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])).astype(int)
    df['open_above_8ema_flag'] = (df['open'] > df['ema_8']).astype(int)
    df['ema_8_21_spread_pct'] = (df['ema_8'] - df['ema_21']) / df['ema_21']
    df['ema_21_50_spread_pct'] = (df['ema_21'] - df['ema_50']) / df['ema_50']
    df['is_low_vol_day'] = (df['volume'] < df['vol_sma_20']).astype(int)
    df['days_below_avg_vol_5d'] = df['is_low_vol_day'].rolling(window=5).sum()
    
    # General volatility and gap features (exact names required by merge_features.py)
    df['vol_contraction_ratio'] = df['volume'] / df['vol_sma_20']
    df['pdh'] = df['high'].shift(1)
    
    # Safe division to prevent inf values
    df['breakout_gap_pct'] = np.where(df['pdh'] != 0, (df['open'] - df['pdh']) / df['pdh'], 0.0)
    
    df.drop(columns=['is_low_vol_day'], inplace=True)
    df.dropna(inplace=True)
    
    return df

def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates 5-minute intraday micro-structure features (VWAP, Time-of-Day R-Vol).
    """
    df = df.copy()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # 1. EMAs for Intraday momentum
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['price_to_8ema_delta'] = (df['close'] - df['ema_8']) / df['ema_8']
    
    # Time-of-day R-Vol
    df['time'] = df.index.time
    df['hist_time_vol'] = df.groupby('time')['volume'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    
    df['intraday_rvol'] = np.where(
        (df['hist_time_vol'] > 0) & (df['hist_time_vol'].notna()), 
        df['volume'] / df['hist_time_vol'], 
        1.0
    )
    
    # VWAP and VWAP stretch calculation
    df['trade_date'] = df.index.date
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['typical_price'] * df['volume']
    
    df['cum_pv'] = df.groupby('trade_date')['pv'].cumsum()
    df['cum_vol'] = df.groupby('trade_date')['volume'].cumsum()
    
    df['vwap'] = np.where(df['cum_vol'] > 0, df['cum_pv'] / df['cum_vol'], df['close'])
    df['vwap_stretch_pct'] = np.where(df['vwap'] > 0, ((df['close'] - df['vwap']) / df['vwap']) * 100, 0.0)
    
    df['intraday_vol_contraction_ratio'] = 1.0 
    
    cols_to_drop = ['time', 'hist_time_vol', 'trade_date', 'typical_price', 'pv', 'cum_pv', 'cum_vol', 'vwap']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    df.dropna(inplace=True)
    return df