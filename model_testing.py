import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


with open('trained_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

X_test = pd.read_csv('test/X_test.csv')
y_test = pd.read_csv('test/y_test.csv')

y_pred = loaded_model.predict(X_test)

print(mean_squared_error(y_test, y_pred)**0.5)
print(mean_absolute_error(y_test, y_pred))
