#!/bin/bash

# Создать датасет
bash data_creation.py

# Предобрабатываем данные
bash model_preparation.py

# Обучение модели
bash model_preprocessing.py

# Обучение модели
bash model_preparation.py

# Проверка модели
bash model_testing.py