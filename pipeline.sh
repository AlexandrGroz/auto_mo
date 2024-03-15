#!/bin/bash

# Создать датасет
python data_creation.py

# Предобрабатываем данные
python model_preprocessing.py

# Обучение модели
python model_preparation.py

# Проверка модели
python model_testing.py