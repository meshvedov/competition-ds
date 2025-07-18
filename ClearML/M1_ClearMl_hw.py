#!/usr/bin/env python

import argparse
import os
import yaml
from pathlib import Path

from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from catboost import CatBoostClassifier

from torch.utils.tensorboard import SummaryWriter
from clearml import Task, Logger

from getpass import getpass


def seed_everything(seed=2024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def check_clearml_env():
    os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml/'
    os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
    os.environ['CLEARML_FILES_HOST'] ='https://files.clear.ml'
    if os.getenv('CLEARML_API_ACCESS_KEY') is None:
        os.environ['CLEARML_API_ACCESS_KEY'] = getpass(prompt="Введите API Access токен: ")
    if os.getenv('CLEARML_API_SECRET_KEY') is None:
        os.environ['CLEARML_API_SECRET_KEY'] = getpass(prompt="Введите API Secret токен: ")
        
# def load_config():
#     config_path = Path(__file__).parent / 'config.yaml'
#     if not config_path.exists():
#         raise FileNotFoundError(f"Configuration file {config_path} not found.")
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#     return config
    
def main():
    try:
        task = Task.init(project_name="ClearMl_logging_from_py", task_name="CatBoost model baseline")
    except:
        check_clearml_env()
        task = Task.init(project_name="ClearMl_logging_from_py", task_name="CatBoost model baseline")    

    task.add_tags(["catboost", "classification", "baseline"])
    logger = Logger.current_logger()
    
    # Парсим аргументы командной строки: --iterations, --verbose
    parser = argparse.ArgumentParser(description="CatBoost model training with ClearML logging")
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for CatBoost')
    parser.add_argument('--verbose', type=int, default=100, help='Verbosity level for CatBoost training')
    args = parser.parse_args()
    
    cb_params = {
        "iterations": args.iterations,
        "verbose": args.verbose,
        "depth": 4,
        "learning_rate": 0.06,
        "loss_function": "MultiClass",
        "custom_metric": ["Recall"],
        # Главная фишка катбуста - работа с категориальными признаками
        "cat_features": cat_features,
        # Регуляризация и ускорение
        "colsample_bylevel": 0.098,
        "subsample": 0.95,
        "l2_leaf_reg": 9,
        "min_data_in_leaf": 243,
        "max_bin": 187,
        "random_strength": 1,
        # Параметры ускорения
        "task_type": "CPU",
        "thread_count": -1,
        "bootstrap_type": "Bernoulli",
        # Важное!
        "random_seed": 2024,
        "early_stopping_rounds": 50,
    }

    task.connect(cb_params, name="CatBoost basic parameters")  # сохраняем словарь с параметрами эксперимента в ClearML
    # Подгружаем данные
    url = "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/quickstart_train.csv"
    rides_info = pd.read_csv(url)

    # Препроцессинг данных
    cat_features = ['model', 'car_type', 'fuel_type']
    targets = ['target_class', 'target_reg']
    features2drop = ['car_id']
    filtered_features = [col for col in rides_info.columns if col not in features2drop + targets]
    num_features = [col for col in filtered_features if col not in cat_features]

    for col in cat_features:
        rides_info[col] = rides_info[col].astype(str)

    # Разделение на тренировочную и валидационную выборки
    train, test = train_test_split(rides_info, test_size=0.2, random_state=42) 
    X_train = train[filtered_features]
    y_train = train['target_class']

    X_test = test[filtered_features]
    y_test = test['target_class'] 

    # Логирование только валидационной выборки
    logger.report_table(title="Valid data", series="datasets", table_plot=test)
    
    #EDA - баланс классов
    rides_info.target_class.value_counts().plot(kind='bar', title='Target class distribution')
    # Сохраняем график в ClearML
    logger.report_matplotlib_figure(
        title="Target class distribution",
        series="target_class_distribution",
        figure=plt
    )

    model = CatBoostClassifier(**cb_params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

    # Сохранение обученной модели
    model.save_model("catboost_model.cbm", format="cbm")

    # Расчет и сохранение метрики на валидационной выборке
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    cls_report = pd.DataFrame(cls_report).transpose()
    logger.report_single_value(name='Accuracy', value=accuracy)
    logger.report_table(title="Classification Report", series="metrics", table_plot=cls_report)

    task.close()

if __name__ == "__main__":
    seed_everything()
    main()