import pandas as pd
import pickle
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV


base_elastic_model = ElasticNet()

param_grid = {'alpha': [0.1, 1, 5, 10, 50, 100],
              'l1_ratio': [.1, .5, .7, .9, .95, .99, 1]}

grid_model = GridSearchCV(estimator=base_elastic_model,
                          param_grid=param_grid,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=2)

X_train = pd.read_csv('train/X_train.csv')
y_train = pd.read_csv('train/y_train.csv')

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    grid_model.fit(X_train, y_train)

# Сохранение модели
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(grid_model, f)
