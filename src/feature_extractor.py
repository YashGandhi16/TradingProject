import pandas as pd

def calculate_strategy_indicators(csv_file_path):
    """
    Loads historical stock data and calculates the EMAs and Volume SMA 
    required for the EMA Trend Continuation strategy.
    """
    print(f"Loading data from {csv_file_path}...")
    
    # 1. Load the data
    # We assume your raw data has standard headers: Date, Open, High, Low, Close, Volume
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file {csv_file_path}. Please check the path.")
        return None

    # 2. Clean and sort the data
    # AI and moving averages strictly require chronological order (oldest to newest)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 3. Calculate 20-day Volume Average (Simple Moving Average)
    # .rolling(window=20).mean() looks back at the last 20 rows and averages them
    df['Vol_20_SMA'] = df['Volume'].rolling(window=20).mean()

    # 4. Calculate Exponential Moving Averages (EMA) for Price
    # adjust=False ensures the math matches how trading platforms (like ThinkOrSwim/TradingView) calculate EMA
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    print("Indicators calculated successfully.")
    return df


# ==========================================
# Execution Block (Where you run your code)
# ==========================================
if __name__ == "__main__":
    # To use this, you need a raw CSV of standard daily stock data (OHLCV)
    # Replace 'raw_apple_data.csv' with your actual file name
    file_name = 'raw_apple_data.csv'
    
    # Run the function
    processed_data = calculate_strategy_indicators(file_name)
    
    if processed_data is not None:
        # Print the last 5 rows to verify the math worked!
        print("\n--- Recent Data with New Indicators ---")
        # We only print specific columns so it doesn't clutter the terminal
        print(processed_data[['Date', 'Close', 'EMA_8', 'EMA_21', 'EMA_50', 'Volume', 'Vol_20_SMA']].tail())
        
        # Optional: Save this new data to a fresh CSV for your AI to read later
        # processed_data.to_csv("apple_with_indicators.csv", index=False)