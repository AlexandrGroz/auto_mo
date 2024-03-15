import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка данных из файла или получение их с помощью API
btc_df_train = pd.read_csv('train/bitcoin_prices_train.csv')
btc_df_test = pd.read_csv('test/bitcoin_prices_test.csv')

X_train = btc_df_train.drop('volume', axis=1)
X_test = btc_df_test.drop('volume', axis=1)
y_train = btc_df_train['volume']
y_test = btc_df_test['volume']

# Масштабирование данных (SCALE)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# Сохранение новых данных в папки train и test
columns = ['open', 'high', 'low', 'close', 'year', 'month', 'day']

pd.DataFrame(X_train, columns=columns).to_csv('train/preprocessing_bitcoin_prices_train.csv', index=False)
pd.DataFrame(X_test, columns=columns).to_csv('test/preprocessing_bitcoin_prices_test.csv', index=False)