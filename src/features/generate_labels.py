import numpy as np
import pandas as pd
from numba import njit

@njit
def apply_triple_barrier(prices: np.ndarray, tp_pct: float, sl_pct: float, max_bars: int) -> np.ndarray:
    """
    O(1) compiled Triple Barrier labeling.
    1 = Take Profit hit first (Successful Breakout)
    0 = Stop Loss hit first OR Time expired (Failed Breakout)
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        # If we are too close to the end of the dataset to look forward, default to 0
        if i + max_bars >= n:
            labels[i] = 0
            continue
            
        entry_price = prices[i]
        
        # Calculate absolute price barriers based on percentages
        # Note: If shorting, you would invert the + and - logic
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)
        
        # Look forward into the future window
        for j in range(1, max_bars + 1):
            future_price = prices[i + j]
            
            # Upper barrier hit
            if future_price >= tp_price:
                labels[i] = 1
                break
            
            # Lower barrier hit
            elif future_price <= sl_price:
                labels[i] = 0
                break
                
        # If the loop finishes without breaking, it hit the Vertical Time Barrier.
        # The default label is already 0, so no action is needed.
        
    return labels

def generate_targets(df: pd.DataFrame, take_profit: float = 0.02, stop_loss: float = 0.01, hold_time_bars: int = 24) -> pd.DataFrame:
    """
    Wraps the Numba function and attaches the target 'y' back to the DataFrame.
    Default parameters: +2% Take Profit, -1% Stop Loss, max hold time of 24 bars (2 hours).
    """
    df = df.copy()
    close_vals = df['close'].to_numpy()
    
    df['target'] = apply_triple_barrier(
        prices=close_vals,
        tp_pct=take_profit,
        sl_pct=stop_loss,
        max_bars=hold_time_bars
    )
    
    return df