import json
import os
import sys
import logging
import json_log_formatter

import mlflow

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold

from xgboost import XGBRegressor

class MLHandler:
    def __init__(self) -> None:
        mlflow.set_tracking_uri("")
        self.mf_client = mlflow.tracking.MlflowClient()
        self.logger = self._define_logger(logger_name="MLLogger")
    
    @staticmethod
    def _define_logger(logger_name: str) -> logging.Logger:
        """ Defines and formats the Logger object with 
            JSON-log-formatter
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