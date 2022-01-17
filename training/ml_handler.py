import json
import os
import sys
import logging
from typing import Tuple
import json_log_formatter

import mlflow

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold

from xgboost import XGBRegressor

class MLHandler:
    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("http://mlflow.aast-innovation.iolap.com")
        mlflow.set_experiment(self.experiment_name)
        self.mf_client = mlflow.tracking.MlflowClient()
        self.logger = self._define_logger(logger_name="MLLogger")
    
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
    
    @staticmethod
    def _get_metrics(pred: np.ndarray, target: np.ndarray, metrics_dict: dict) -> dict:
        """ Performs metric calcuations for predictions in respect to the target values 
            :argument: pred - Numpy array containing Model predictions
            :argument: target - Numpy array containing real values
            :argument: metrics_dict - Empty dictionary
            :return: metrics_dict - Dictionary with all calculated metrics
        """
        mse = mean_squared_error(target, pred)
        metrics_dict['MSE'] = mse.round(2)
        rmse = np.sqrt(mse)
        metrics_dict['RMSE'] = rmse.round(2)
        r2 = r2_score(target, pred)
        metrics_dict['R2'] = r2.round(2)
        mae = mean_absolute_error(target, pred)
        metrics_dict['MAE'] = mae.round(2)
        return metrics_dict
    
    @staticmethod
    def plot_residuals(pred: np.ndarray, real: np.ndarray) -> plt.figure:
        """ Plots the residual errors between the real and predicted value 
            :argument: pred - Numpy array containing Model predictions
            :argument: real - Numpy array containing real values
            :return: fig - Matplotlib figure object
        """
        # Calculate residual errors
        residuals = real - pred
        fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(20, 8))
        # Residuals Errors plot
        axs[0].scatter(x=pred, y=residuals, color='g', alpha=0.7)
        axs[0].axhline(y=0, linestyle='--', color='black', linewidth=3.5)
        axs[0].set_xlabel('Estimations', fontsize=17)
        axs[0].set_ylabel('Residuals', fontsize=17)
        axs[0].set_title('Estimation residuals', fontsize=18)
        axs[0].tick_params(axis='both', which='major', labelsize=17)
        axs[0].grid()
        # Estimation Errors plot
        axs[1].scatter(x=pred, y=real, c='orange', alpha=0.7)
        axs[1].plot([0, 1], [0, 1], transform=axs[1].transAxes, ls="--", c=".1", linewidth=3, label='Best fit')
        axs[1].set_xlabel('Estimations', fontsize=17)
        axs[1].set_ylabel('Real values', fontsize=17)
        axs[1].set_title('Estimation error', fontsize=18)
        axs[1].tick_params(axis='both', which='major', labelsize=17)
        axs[1].set_xlim(min(real), max(real))
        axs[1].set_ylim(min(real), max(real))
        axs[1].grid()
        axs[1].legend(loc='best', fontsize=17)
        return fig
        
    
    def define_ml_dataset(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                          features: list) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """ Creates the training and test datasets to train the models 
            :argument: train_data - Pandas DataFrame as training dataset, should contain target RUL
            :argument: test_data - Pandas DataFrame as test dataset, should contain target RUL
            :argument: features - List containing name of columns to use as features from training set
        """
        self.logger.info("Defining the ML dataset...")
        # Get features and label for training dataset
        x_train = train_data[features]
        y_train = train_data['rul']
        # Get only the last row for each of the units/engines because we only have labels for those rows
        test_grouped = test_data.groupby('unit').last().reset_index()
        # Get features and label for test dataset
        x_test = test_grouped[features]
        y_test = test_grouped['rul']
        self.logger.info("Defined the ML dataset!")
        return x_train, y_train, x_test, y_test
    
    def evaluate_model(self, model, x_train, y_train, x_test, y_test) -> Tuple[dict, dict, plt.figure]:
        """ Evaluates model on both trainig and testing data """
        self.logger.info("Evaluating the model...")
        # Predict for both datasets
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)
        # Calculate metrics
        train_metrics, test_metrics = {'data': 'training'}, {'data': 'test'}
        # Get training metrics
        train_metrics = self._get_metrics(train_predict, y_train, train_metrics)
        # Get test metrics
        test_metrics = self._get_metrics(test_predict, y_test, test_metrics)
        # Plot the residuals error
        self.logger.info("Plotting the residuals...")
        figure = self.plot_residuals(test_predict, y_test)
        self.logger.info("Train metrics", extra=train_metrics)
        self.logger.info("Test metrics", extra=test_metrics)
        return train_metrics, test_metrics, figure
    
    def train_xgboost(self, x_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> Tuple[XGBRegressor, pd.DataFrame]:
        """ Trains the XGBoost Regression model on given training dataset and performs GridSearchCV on given parameters dict
            :argument: x_train - Pandas DataFrame as training dataset, does not contain target RUL
            :argument: y_train - Pandas Series, contain target RUL
            :argument: parameters - Dictionary with XGBoostRegressor parameters to perform GridSearch over
        """
        self.logger.info("Training the XGBRegressor model...")
        # Instantiate the model object
        model = XGBRegressor(objective="reg:squarederror", random_state=123, booster='gbtree')
        # Define the group data folding with column 'unit'
        group_fold = GroupKFold(n_splits=3)
        # Define the GridSearchCV
        grid_search = GridSearchCV(model, param_grid=parameters, n_jobs=-1,
                                cv=group_fold.split(x_train, groups=x_train['unit']),
                                verbose=3, error_score='raise', scoring='neg_mean_squared_error')
        # Train the model with GridSearch
        grid_search.fit(x_train, y_train)
        # Grid Search results
        grid_results = pd.DataFrame(grid_search.cv_results_)
        # Get best model from GridSearch
        best_model = grid_search.best_estimator_
        # Print best score and best params
        best_score = abs(grid_search.best_score_)
        self.logger.info(f'Best model scores', extra={"MSE": best_score.round(2), "RMSE": np.sqrt(best_score).round(2)})
        self.logger.info(f"Best model parameters", extra=grid_search.best_params_)
        self.logger.info("XGBRegressor model finished training!")
        return best_model, grid_results