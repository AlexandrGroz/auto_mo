#!/bin/bash

# Создать датасет
python3 data_creation.py

# Предобрабатываем данные
python3 model_preprocessing.py

# Обучение модели
python3 model_preparation.py

# Проверка модели
python3 model_testing.py