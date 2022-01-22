from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def load_raw_data(txt_filename: str) -> pd.DataFrame:
    """ Loads the .txt data from the Data folder as pandas DataFrame 
        :argument: txt_filename - name of the text file with .txt extension
        :return: dataframe - pandas DataFrame
    """
    # Define the data folder path
    data_path = Path().absolute().parent.joinpath('data', 'raw')
    # Open with pandas without header and space as separator
    train = pd.read_csv(data_path.joinpath(txt_filename), sep='\s+', header=None).dropna(axis=1, how='all')
    test = pd.read_csv(data_path.joinpath(txt_filename.replace('train', 'test')), sep='\s+', header=None).dropna(axis=1, how='all')
    y_test = pd.read_csv(data_path.joinpath(txt_filename.replace('train', 'RUL')), sep='\s+', header=None, names=['rul']).dropna(axis=1, how='all')
    # Define number of sensor columns
    sensors_number = len(train.columns) - 5
    # Rename the columns to corrensponding value
    column_names = ['unit', 'cycle', 'altitude', 'mach', 'tra'] + [f'sensor_{i}' for i in range(1, sensors_number + 1)]
    train.columns = column_names
    test.columns = column_names
    return train, test, y_test

def create_target(raw_data: pd.DataFrame) -> pd.DataFrame:
    """ Creates the RUL target variable based on max cycles from the dataset 
        :argument: raw_data - Pandas DataFrame containing training data
        :return: dataset - Pandas DataFrame containing training data and target variable
    """
    data = raw_data.copy()
    # Group the data by unit column and calculate the max cycle
    grouped = data.groupby('unit')
    max_cycle = grouped['cycle'].max()
    # Merge the max cycle back to the data
    data = data.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit', right_index=True)
    # Calculate difference between max cycle and current cycle, create RUL
    data['rul'] = data['max_cycle'] - data['cycle']
    # Drop the max cycle column
    data.drop('max_cycle', axis=1, inplace=True)
    return data

def round_conditions(input_data: pd.DataFrame) -> pd.DataFrame:
    """ Rounds the values of condition columns (altitude, mach, tra) """
    data = input_data.copy()
    data['altitude'] = data['altitude'].round()
    data['mach'] = data['mach'].round(2)
    data['tra'] = data['tra'].round()
    # Concatenate all 3 conditions into 1
    data['condition'] = data['altitude'] + data['mach'] + data['tra']
    keys = data['condition'].unique()
    mapping = {k: v for k, v in zip(keys, range(1, len(keys) + 1))}
    data['condition'] = data['condition'].map(mapping)
    return data

def standardize(input_data: pd.DataFrame):
    """ Standardizes the sensor values based on condition to have same mean to be comparable """
    data = input_data.copy()
    sensors = [e for e in list(data.columns) if 'sensor_' in e]
    for condition in data['condition'].unique():
        for column in sensors:
            mean =  data.loc[data['condition'] == condition, column].mean()
            std = data.loc[data['condition'] == condition, column].std()
            data.loc[data['condition'] == condition,column] = data.loc[data['condition'] == condition, column].map(lambda x: (x - mean) / (std + 0.0000001))
    return data

def get_condition_stats(data: pd.DataFrame) -> pd.DataFrame:
    """ Computes the Mean and Std for every sensor based on certain condition """
    sensors = [e for e in list(data.columns) if 'sensor_' in e]
    means, stds, conditions = [], [], []
    sensor_names = []
    for condition in data['condition'].unique():
        for column in sensors:
            sensor_names.append(column)
            conditions.append(condition)
            means.append(data.loc[data['condition'] == condition, column].mean())
            stds.append(data.loc[data['condition'] == condition, column].std())
    stats = pd.DataFrame(list(zip(sensor_names, means, stds, conditions)), columns=['sensor_name','mean', 'std', 'condition'])
    return stats

def smooth_data(input_data: pd.DataFrame, window: int) -> pd.DataFrame:
    """ Smooths the sensor measurements with Moving Average and specified window 
        :argument: input_data - Pandas Dataframe containing data
        :argument: window - Integer representing the moving average window size
        :return: smoothed_data - Pandas Dataframe with smoothed sensor measurements
    """
    smoothed_data = input_data.copy()
    sensors = [e for e in list(smoothed_data.columns) if 'sensor_' in e]
    smoothed_data[sensors] = smoothed_data.groupby('unit')[sensors].apply(lambda column: column.rolling(window=window, min_periods=1).mean())
    return smoothed_data

def plot_residuals(pred, real) -> None:
    """ Plots the residual errors between the real and predicted value """
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
    plt.show()
    return fig