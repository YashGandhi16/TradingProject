import pandas as pd

def align_timeframes(intraday_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely merges daily macro features onto 5-minute intraday data.
    Prevents forward-looking data leakage by shifting EOD daily features by 1 day.
    """
    intraday_df = intraday_df.copy()
    macro_df = macro_df.copy()
    
    # 1. Save the original datetime index safely into a column to survive the merge
    intraday_df['temp_datetime_idx'] = intraday_df.index
    
    # Extract the pure date (00:00:00) to act as the exact merge key
    intraday_df['trade_date'] = intraday_df.index.normalize()
    macro_df['trade_date'] = macro_df.index.normalize()
    
    # 2. Categorize Macro Features
    # EOD features need the daily close to calculate, so they must be shifted to T+1
    eod_columns = ['ema_8', 'ema_21', 'ema_50', 'vol_sma_20', 'vol_contraction_ratio']
    
    # Morning features are known at 9:30 AM (based on yesterday's close/today's open)
    morning_columns = ['pdh', 'breakout_gap_pct']
    
    # 3. Shift ONLY the End-of-Day features
    # Because macro_df only contains valid trading days, shift(1) safely skips weekends/holidays
    macro_eod_shifted = macro_df[eod_columns].shift(1)
    macro_morning = macro_df[morning_columns]
    
    # Recombine the safe daily dataset
    safe_macro_df = pd.concat([macro_eod_shifted, macro_morning], axis=1)
    safe_macro_df['trade_date'] = macro_df['trade_date']
    
    # Add prefix so the ML model and feature importance charts are clearly labeled
    safe_macro_df = safe_macro_df.add_prefix('macro_')
    
    # 4. Merge onto the 5-minute Intraday dataset
    merged_df = intraday_df.merge(
        safe_macro_df,
        left_on='trade_date',
        right_on='macro_trade_date',
        how='left'
    )
    
    # Clean up routing keys and drop the NaN rows created by the macro shift
    merged_df.drop(columns=['trade_date', 'macro_trade_date'], inplace=True)
    merged_df.dropna(inplace=True)
    
    # 5. Restore the original 5-minute datetime index
    merged_df.set_index('temp_datetime_idx', inplace=True)
    merged_df.index.name = 'date'
    
    return merged_df