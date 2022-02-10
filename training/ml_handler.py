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
from sklearn.preprocessing import StandardScaler
import pickle

from xgboost import XGBRegressor
from xgboost import plot_importance

class MLHandler:
    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("http://mlops-mlflow-load-balancer-1002257987.us-east-1.elb.amazonaws.com")
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
        if logger.hasHandlers():
            logger.handlers.clear()
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
        axs[0].set_xlabel('Estimations (RUL)', fontsize=17)
        axs[0].set_ylabel('Residuals', fontsize=17)
        axs[0].set_title('Estimation residuals', fontsize=18)
        axs[0].tick_params(axis='both', which='major', labelsize=17)
        axs[0].grid()
        # Estimation Errors plot
        axs[1].scatter(x=pred, y=real, c='orange', alpha=0.7)
        axs[1].plot([0, 1], [0, 1], transform=axs[1].transAxes, ls="--", c=".1", linewidth=3, label='Best fit')
        axs[1].set_xlabel('Estimations (RUL)', fontsize=17)
        axs[1].set_ylabel('Real values', fontsize=17)
        axs[1].set_title('Estimation error', fontsize=18)
        axs[1].tick_params(axis='both', which='major', labelsize=17)
        axs[1].set_xlim(min(real), max(real))
        axs[1].set_ylim(min(real), max(real))
        axs[1].grid()
        axs[1].legend(loc='best', fontsize=17)
        return fig
    
    def define_ml_dataset(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                          features: list, run: mlflow.entities.run) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """ Creates the training and test datasets to train the models 
            :argument: train_data - Pandas DataFrame as training dataset, should contain target RUL
            :argument: test_data - Pandas DataFrame as test dataset, should contain target RUL
            :argument: features - List containing name of columns to use as features from training set
        """
        self.logger.info("Defining the ML dataset...")
        train_data = train_data.astype('float64')
        test_data = test_data.astype('float64')
        # Get features and label for training dataset
        x_train = train_data[features]
        y_train = train_data['rul']
        # Get only the last row for each of the units/engines because we only have labels for those rows
        test_grouped = test_data.groupby('unit').last().reset_index()
        # Get features and label for test dataset
        x_test = test_grouped[features]
        y_test = test_grouped['rul']
        self.logger.info("Defined the ML dataset!")
        # Save the features list used for training in MLflow
        features_used = {'Features': features}
        self.mf_client.log_dict(run.info.run_id, features_used, 'Features/features.json')
        return x_train, y_train, x_test, y_test
    
    def standardize_with_scaler(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame):
        """ Standardizes the sensor values based on condition to have same mean to be comparable on both
             training and testing data with StandardScaler
        """
        train_data = train_dataset.copy()
        test_data = test_dataset.copy()
        self.logger.info("Scaling the dataset based on condition...")
        standard_scaler = StandardScaler()
        sensors = [e for e in list(train_data.columns) if 'sensor_' in e]
        for condition in train_data['condition'].unique():
            standard_scaler.fit(train_data.loc[train_data['condition'] == condition, sensors])
            train_data.loc[train_data['condition'] == condition, sensors] = standard_scaler.transform(train_data.loc[train_data['condition'] == condition, sensors])
            test_data.loc[test_data['condition'] == condition, sensors] = standard_scaler.transform(test_data.loc[test_data['condition'] == condition, sensors])
        pickle.dump(standard_scaler, open('StandardScaler.pkl', 'wb')) 
        mlflow.log_artifact("StandardScaler.pkl", "StandardScaler")
        os.remove('StandardScaler.pkl')
        return train_data, test_data
    
    def evaluate_model(self, model, x_train, y_train, x_test, y_test, run: mlflow.entities.run) -> Tuple[dict, dict, plt.figure]:
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
        figure_test = self.plot_residuals(test_predict, y_test)
        self.logger.info("Train metrics", extra=train_metrics)
        self.logger.info("Test metrics", extra=test_metrics)
        # Save metrics to the MLflow
        for metric_name, value in test_metrics.items():
            if metric_name == "data":
                continue
            else:
                mlflow.log_metric(metric_name, value)
        self.mf_client.log_dict(run.info.run_id, train_metrics, 'Metrics/train.json')
        self.mf_client.log_dict(run.info.run_id, test_metrics, 'Metrics/test.json')
        # Save the residuals plot in MLflow Artifacts
        self.mf_client.log_figure(run.info.run_id, figure_test, "Plots/test_residual_plot.png")
        return train_metrics, test_metrics
    
    def train_xgboost(self, x_train: pd.DataFrame, y_train: pd.Series, parameters: dict, 
                      run: mlflow.entities.run, group_fold_data: pd.DataFrame) -> Tuple[XGBRegressor, pd.DataFrame]:
        """ Trains the XGBoost Regression model on given training dataset and performs GridSearchCV on given parameters dict
            :argument: x_train - Pandas DataFrame as training dataset, does not contain target RUL
            :argument: y_train - Pandas Series, contain target RUL
            :argument: parameters - Dictionary with XGBoostRegressor parameters to perform GridSearch over
        """
        self.logger.info("Training the XGBRegressor model...")
        # Instantiate the model object
        model = XGBRegressor(objective="reg:squarederror", random_state=123, booster='gbtree')
        # Define the group data folding with column 'unit'
        group_fold = GroupKFold(n_splits=2)
        # Define the GridSearchCV
        grid_search = GridSearchCV(model, param_grid=parameters, n_jobs=-1,
                                cv=group_fold.split(group_fold_data, groups=group_fold_data['unit']),
                                verbose=3, error_score='raise', scoring='neg_mean_squared_error')
        # Train the model with GridSearch
        grid_search.fit(x_train, y_train)
        # Grid Search results
        grid_results = pd.DataFrame(grid_search.cv_results_)
        # Get best model from GridSearch
        best_model = grid_search.best_estimator_
        # Log best model in MLflow
        mlflow_model_info = mlflow.sklearn.log_model(best_model, "XGBRegressor")
        # Log the best parameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        # Save feature importance in MLflow
        importance_figure, ax = plt.subplots(nrows=1, ncols=1)
        plot_importance(best_model, ax=ax)
        self.mf_client.log_figure(run.info.run_id, importance_figure, "Plots/feature_importance.png")
        # Print best score and best params
        best_score = abs(grid_search.best_score_)
        valid_score = {"MSE": best_score.round(2), "RMSE": np.sqrt(best_score).round(2)}
        self.mf_client.log_dict(run.info.run_id, valid_score, 'Metrics/validation.json')
        self.logger.info(f'Best model scores', extra=valid_score)
        self.logger.info(f"Best model parameters", extra=grid_search.best_params_)
        self.logger.info("GridSearchCV for XGBRegressor finished!")
        return best_model, grid_results