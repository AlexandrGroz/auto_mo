import ccxt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import os

def generate_random_timestamp(start_year=2015, end_year=2023):
    start_timestamp = int(pd.Timestamp(f'{start_year}-01-01').timestamp() * 1000)
    end_timestamp = int(pd.Timestamp(f'{end_year}-01-01').timestamp() * 1000)
    return random.randint(start_timestamp, end_timestamp)

def fetch_btc_data(since_timestamp):
    binance = ccxt.binance()
    btc_prices = binance.fetch_ohlcv('BTC/USDT', timeframe='1d', since=since_timestamp)
    btc_df = pd.DataFrame(btc_prices, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
    btc_df['year'] = btc_df['timestamp'].dt.year
    btc_df['month'] = btc_df['timestamp'].dt.month
    btc_df['day'] = btc_df['timestamp'].dt.day
    btc_df.drop(columns=['timestamp'], inplace=True)
    return btc_df

def split_and_save_data(df, train_path='train/bitcoin_prices_train.csv', test_path='test/bitcoin_prices_test.csv', test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=True, random_state=random_state)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

def main():
    since_timestamp = generate_random_timestamp()
    btc_df = fetch_btc_data(since_timestamp)
    split_and_save_data(btc_df)

if __name__ == "__main__":
    main()