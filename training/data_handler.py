import logging
import json_log_formatter
import pandas as pd
import numpy as np
import sys

import awswrangler

class DataHandler:
    def __init__(self, s3_bucket: str) -> None:
        self.bucket = s3_bucket
        self.logger = self._define_logger(logger_name="DataLogger")
    
    @staticmethod
    def _define_logger(logger_name: str) -> logging.Logger:
        """ Defines and formats the Logger object with JSON-log-formatter
            :argument: logger_name - Name for the logger object
            :return: logger - Logger object
        """
        # Define the Formatter
        formatter = json_log_formatter.JSONFormatter()
        json_handler = logging.StreamHandler(sys.stdout)
        json_handler.setFormatter(formatter)
        # Define the Logger
        logger = logging.getLogger(logger_name)
        logger.addHandler(json_handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """ Loads the dataset from the S3 bucket based on the file_name
            :argument: file_path - Path to the file stored in S3 bucket
        """
        self.logger.info("Loading the data from S3...")
        # Get the parquet data
        data = awswrangler.s3.read_parquet(path=[f"s3://{self.bucket}/{file_path}"])
        self.logger.info("Data loaded!")
        return data
    
    def define_conditions(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """ Rounds the values of condition columns (altitude, mach, tra) """
        self.logger.info("Defining the conditions in data...")
        data = input_data.copy()
        data['altitude'] = data['altitude'].round()
        data['mach'] = data['mach'].round(2)
        data['tra'] = data['tra'].round()
        # Concatenate all 3 conditions into 1
        data['condition'] = data['altitude'] + data['mach'] + data['tra']
        keys = data['condition'].unique()
        mapping = {k: v for k, v in zip(keys, range(1, len(keys) + 1))}
        data['condition'] = data['condition'].map(mapping)
        self.logger.info("Conditions successfully defined!")
        return data

    def standardize(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """ Standardizes the sensor values based on condition to have same mean to be comparable """
        self.logger.info("Standardizing data...")
        data = input_data.copy()
        sensors = [e for e in list(data.columns) if 'sensor_' in e]
        for condition in data['condition'].unique():
            for column in sensors:
                mean =  data.loc[data['condition'] == condition, column].mean()
                std = data.loc[data['condition'] == condition, column].std()
                data.loc[data['condition'] == condition,column] = data.loc[data['condition'] == condition, column].map(lambda x: (x - mean) / (std + 0.0000001))
        self.logger.info("Data successfully standardized!")
        return data
    
    def smooth_data(self, input_data: pd.DataFrame, window: int) -> pd.DataFrame:
        """ Smooths the sensor measurements with Moving Average and specified window 
            :argument: input_data - Pandas Dataframe containing data
            :argument: window - Integer representing the moving average window size
            :return: smoothed_data - Pandas Dataframe with smoothed sensor measurements
        """
        self.logger.info("Smoothing the data...")
        smoothed_data = input_data.copy()
        sensors = [e for e in list(smoothed_data.columns) if 'sensor_' in e]
        smoothed_data[sensors] = smoothed_data.groupby('unit')[sensors].apply(lambda column: column.rolling(window=window, min_periods=1).mean())
        self.logger.info("Data successfully smoothed!")
        return smoothed_data
    