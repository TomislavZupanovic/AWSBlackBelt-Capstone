from cgi import test
import os
import logging
import json_log_formatter

import mlflow
import pandas as pd
import numpy as np
from data_handler import DataHandler
from ml_handler import MLHandler
from datetime import datetime


if __name__ == '__main__':
    # Instantiate DataHandler and MLHandler
    data_handler = DataHandler(s3_bucket="mlops-storage-bucket")
    ml_handler = MLHandler(experiment_name="Experimentation-Phase")
    # Get training and testing data S3 paths
    train_path = os.environ.get('TrainDataPath', None)
    test_path = os.environ.get('TestDataPath', None)
    # If paths are not specified
    if not train_path:
        # Use current date's data in S3 bucket
        current_date = datetime.now().strftime('%Y-%m-%d')
        train_path = f"curated/total/parquet/{current_date}-train.parquet"
        test_path = f"curated/total/parquet/{current_date}-test.parquet"
    # Load both train and test data
    train_data = data_handler.load_data(train_path)
    test_data = data_handler.load_data(test_path)
    # Perform condition defining
    train_data = data_handler.define_conditions(train_data)
    test_data = data_handler.define_conditions(test_data)
    # Standardize the data
    standardized_train_data = data_handler.standardize(train_data)
    standardized_test_data = data_handler.standardize(test_data)
    # Smooth the data
    smoothed_train_data = data_handler.smooth_data(standardized_train_data, window=10)
    smoothed_test_data = data_handler.smooth_data(standardized_test_data, window=10)
    # Get train, test and target split
    selected_features = ['unit', 'altitude', 'mach', 'tra', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 'sensor_7',
                    'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
                    'sensor_17', 'sensor_20', 'sensor_21']
    x_train, y_train, x_test, y_test = ml_handler.define_ml_dataset(smoothed_train_data, smoothed_test_data,
                                                                    selected_features)
    # TODO: add Mlflow tracking