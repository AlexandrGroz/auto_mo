import ccxt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import os

# Создание объекта для работы с API Binance
binance = ccxt.binance()

# Генерация случайной временной метки since с 2015 по 2023 год
since_timestamp = random.randint(1420070400000, 1704067200000)  # от 1 января 2015 года в миллисекундах до 1 января 2024 года

# Получение исторических данных о ценах на биткоин
btc_prices = binance.fetch_ohlcv('BTC/USDT', timeframe='1d', since=since_timestamp)

# Преобразование данных в DataFrame
btc_df = pd.DataFrame(btc_prices, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Преобразование временных меток
btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')

# Разделение данных на обучающий (train) и тестовый (test) наборы
train_df, test_df = train_test_split(btc_df, test_size=0.2, shuffle=True, random_state=42)

# Создание папок train и test, если они еще не существуют
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Сохранение данных в папки train и test
train_df.to_csv('train/bitcoin_prices_train.csv', index=False)
test_df.to_csv('test/bitcoin_prices_test.csv', index=False)
