import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(stock_data_path, sentiment_data_path):
    """
    Loads and preprocesses stock and sentiment data from CSV files, performs feature engineering,
    and splits the data into training and testing sets.

    Parameters:
    - stock_data_path: Path to the stock data CSV file.
    - sentiment_data_path: Path to the sentiment data CSV file.

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets for training and testing.
    """
    # Load stock data from CSV file
    stock_data = pd.read_csv(stock_data_path, parse_dates=['Date'])
    stock_data.sort_values('Date', inplace=True)
    
    # Load sentiment data from CSV file
    sentiment_data = pd.read_csv(sentiment_data_path, parse_dates=['Date'])
    sentiment_data.sort_values('Date', inplace=True)

    # Remove timezone information from the 'Date' columns
    stock_data['Date'] = stock_data['Date'].dt.tz_localize(None)
    sentiment_data['Date'] = sentiment_data['Date'].dt.tz_localize(None)
    
    # Merge datasets on Date and StockName
    merged_data = pd.merge(stock_data, sentiment_data, on=['Date', 'Stock Name'])
    
    print(f"Merged Data Sample ({len(merged_data)} rows):")
    print(merged_data.head(10))

    # Feature Engineering: Adding moving averages and RSI
    merged_data['SMA_50'] = merged_data['Close'].rolling(window=50).mean()
    merged_data['SMA_200'] = merged_data['Close'].rolling(window=200).mean()
    merged_data['RSI'] = compute_rsi(merged_data['Close'], window=14)
    
    # Handle missing values using ffill
    merged_data.ffill(inplace=True)
    
    # Define features and target
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'Sentiment']
    X = merged_data[feature_columns]
    y = merged_data['Close'].shift(-1) 
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, merged_data

def compute_rsi(series, window=14):
    """
    Computes the Relative Strength Index (RSI) for a given series.

    Parameters:
    - series: Pandas Series of prices.
    - window: The number of periods to use for RSI calculation.

    Returns:
    - Pandas Series representing the RSI.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

if __name__ == "__main__":
    # Paths to stock and sentiment data files
    stock_data_path = './Dataset/stock_yfinance_data.csv'
    sentiment_data_path = './Dataset/processed_sentiment_scores.csv'
    
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, merged_data = load_and_preprocess_data(stock_data_path, sentiment_data_path)
    
    # Save preprocessed data to CSV
    merged_data.to_csv('./Dataset/merged_preprocessed_data.csv', index=False)
    pd.DataFrame(X_train).to_csv('./Dataset/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('./Dataset/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('./Dataset/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('./Dataset/y_test.csv', index=False)
    
    print("Data preprocessing complete and files saved.")
