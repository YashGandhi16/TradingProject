"""
EMA Trend Continuation — Historical Setup Screener
====================================================

Scans daily OHLCV data for a list of tickers and flags dates where all three
conditions of the "EMA Trend Continuation" setup are met:

  1. Uptrend Alignment   : 8 EMA > 21 EMA > 50 EMA (daily)
  2. Contraction          : >= 3 of the last 5 days had volume below the
                            20-day volume SMA
  3. Breakout Trigger     : Today's close > highest high of the prior 10 days
                            AND today's volume > 20-day volume SMA

Output: potential_setups.csv with columns
  Ticker, Setup_Date, Close_Price, Volume

This script only finds CANDIDATE DAILY SETUP DATES. It does not identify the
5-minute intraday breakout timestamp — that step is manual, by design (per
the workflow: screen daily -> confirm on the 5m chart -> log the trade).

Requirements:
    pip install yfinance pandas --break-system-packages   # or in a venv

Run:
    python ema_trend_screener.py
"""

import time
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# 20-30 high-volume tech/growth tickers. Edit freely to match your universe.
TICKERS = [
    "NVDA", "AMD", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX",
    "AVGO", "CRM", "PLTR", "SMCI", "SOFI", "COIN", "MSTR", "RIVN", "SNOW",
    "SHOP", "UBER", "ARM", "MU", "MRVL", "DELL", "ORCL", "NOW", "PANW", 
]

LOOKBACK_PERIOD = "2y"      # yfinance period string
MIN_ROWS_REQUIRED = 90      # need enough history for 50 EMA + 20 SMA to warm up
CONTRACTION_WINDOW = 5      # look at last 5 days for the "low volume" score
CONTRACTION_MIN_DAYS = 3    # need >= 3 of those 5 days below the 20d vol SMA
BREAKOUT_LOOKBACK = 10      # rolling high window for breakout trigger
VOL_SMA_WINDOW = 20         # volume SMA window used in both conditions

OUTPUT_CSV = "data/claude_potential_setups.csv"
REQUEST_PAUSE_SEC = 0.5     # small pause between tickers to be polite to the API


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_ticker_data(ticker: str, period: str = LOOKBACK_PERIOD):
    """
    Download daily OHLCV data for a single ticker.
    Returns None (instead of raising) on any failure so the batch loop
    can skip bad tickers without crashing.
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"  [WARN] {ticker}: download failed ({e})")
        return None

    if df is None or df.empty:
        print(f"  [WARN] {ticker}: no data returned")
        return None

    # yfinance sometimes returns MultiIndex columns even for a single ticker
    # (esp. across versions) — flatten them to plain OHLCV column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  [WARN] {ticker}: missing columns {missing}, skipping")
        return None

    if len(df) < MIN_ROWS_REQUIRED:
        print(f"  [WARN] {ticker}: only {len(df)} rows, not enough history, skipping")
        return None

    # Drop any rows with NaN close/volume (holidays, bad data, etc.)
    df = df.dropna(subset=["Close", "Volume"])

    return df


# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all indicators needed to evaluate the setup:
      - 8 EMA, 21 EMA, 50 EMA on Close
      - 20-day Volume SMA
      - 10-day rolling high of High (using PRIOR 10 days, excludes today)
    """
    df = df.copy()

    df["EMA_8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df["Vol_SMA_20"] = df["Volume"].rolling(window=VOL_SMA_WINDOW).mean()

    # Rolling high of the PRIOR N days (shift(1) so "today" isn't included
    # in its own resistance level — otherwise breakout day would trivially
    # never exceed a high that includes itself).
    df["Rolling_High_10"] = (
        df["High"].shift(1).rolling(window=BREAKOUT_LOOKBACK).max()
    )

    # For the contraction score: was each day's volume below the 20d vol SMA?
    df["Below_Vol_SMA"] = df["Volume"] < df["Vol_SMA_20"]

    # Score = how many of the last CONTRACTION_WINDOW days (including today)
    # were below the 20d vol SMA.
    df["Contraction_Score"] = (
        df["Below_Vol_SMA"].rolling(window=CONTRACTION_WINDOW).sum()
    )

    return df


# ---------------------------------------------------------------------------
# Setup evaluation
# ---------------------------------------------------------------------------

def find_setups(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Applies the three setup conditions row-by-row and returns only the
    rows (dates) where all three are true simultaneously.
    """
    df = add_indicators(df)

    # Condition 1: Uptrend alignment
    cond_uptrend = (df["EMA_8"] > df["EMA_21"]) & (df["EMA_21"] > df["EMA_50"])

    # Condition 2: Contraction — at least 3 of last 5 days below 20d vol SMA
    cond_contraction = df["Contraction_Score"] >= CONTRACTION_MIN_DAYS

    # Condition 3: Breakout — close above prior 10-day high AND volume above 20d SMA
    cond_breakout = (df["Close"] > df["Rolling_High_10"]) & (
        df["Volume"] > df["Vol_SMA_20"]
    )

    all_conditions = cond_uptrend & cond_contraction & cond_breakout

    hits = df.loc[all_conditions].copy()
    if hits.empty:
        return pd.DataFrame(columns=["Ticker", "Setup_Date", "Close_Price", "Volume"])

    result = pd.DataFrame({
        "Ticker": ticker,
        "Setup_Date": hits.index.strftime("%Y-%m-%d"),
        "Close_Price": hits["Close"].round(2).values,
        "Volume": hits["Volume"].astype(int).values,
    })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Scanning {len(TICKERS)} tickers over the last {LOOKBACK_PERIOD}...\n")

    all_results = []

    for i, ticker in enumerate(TICKERS, start=1):
        print(f"[{i}/{len(TICKERS)}] {ticker}...")
        df = download_ticker_data(ticker)

        if df is None:
            continue

        try:
            setups = find_setups(df, ticker)
        except Exception as e:
            print(f"  [WARN] {ticker}: error evaluating setup conditions ({e})")
            continue

        if not setups.empty:
            print(f"  -> {len(setups)} candidate setup date(s) found")
            all_results.append(setups)
        else:
            print("  -> no setups found")

        time.sleep(REQUEST_PAUSE_SEC)

    if not all_results:
        print("\nNo setups found across any ticker. Try loosening thresholds or widening the ticker list.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df.sort_values(["Setup_Date", "Ticker"]).reset_index(drop=True)

    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. {len(final_df)} candidate setup dates found across "
          f"{final_df['Ticker'].nunique()} tickers.")
    print(f"Saved to {OUTPUT_CSV}")
    print("\nNext step: pull up each Ticker/Setup_Date on a 5-minute chart, "
          "confirm the actual intraday breakout candle, and log it manually "
          "into your trade dataset.")


if __name__ == "__main__":
    main()