import mlflow
import numpy as np
import pandas as pd
import logging
import sys
import os
import json
import json_log_formatter
import awswrangler
from sklearn.preprocessing import StandardScaler
import pickle
import boto3

class InferenceHandler:
    def __init__(self, model_name: str, stage: str) -> None:
        self.model_name = model_name
        self.model_version = None
        self.stage = stage
        mlflow.set_tracking_uri("http://mlops-mlflow-load-balancer-1002257987.us-east-1.elb.amazonaws.com")
        self.mf_client = mlflow.tracking.MlflowClient()
        self.logger = self._define_logger(logger_name="InferenceLogger")
        
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
    
    def get_model(self) -> mlflow.pyfunc.PyFuncModel:
        """ Load the saved model from MLflow for given model name and stage 
            :argument: None
            :return: model - MLflow saved model
        """
        self.logger.info("Getting the model...")
        # Check if there are any models with model_name
        try:
            version_info = self.mf_client.get_latest_versions(self.model_name, stages=[self.stage])[0]
        except:
            warning = f"There is no available model for stage {self.stage} with Model Name: {self.model_name}!"
            self.logger.warning(warning)
            raise FileNotFoundError(warning)
        # Load the model into memory
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{self.model_name}/{self.stage}")
        self.model_version = version_info.version
        self.logger.info("Model loaded!")
        return model
    
    def load_scaler(self, run_id: str) -> StandardScaler:
        """ Loads the saved Standard Scaler from the MLflow artifacts, used to scale 
            the sensor values for the incoming inference data
            :argument: run_id = Mlflow run Id for the loaded model
            :return: scaler - Sklearn StandardScaler object
        """
        os.makedirs("downloaded_artifacts", exist_ok=True)
        # Check if scaler exists in MLflow
        try:
            scaler_path = self.mf_client.download_artifacts(run_id, "StandardScaler/StandardScaler.pkl", "downloaded_artifacts")
        except:
            warning = f"There is no saved StandardScaler!"
            self.logger.warning(warning)
            raise FileNotFoundError(warning)
        else:
            # Load the Scaler into memory
            self.logger.info("Loading the StandardScaler...")
            scaler = pickle.load(open(scaler_path, 'rb'))
            # Remove the local file downlaoded by MLflow
            os.remove(scaler_path)
            self.logger.info("StandardScaler successfully loaded!")
        return scaler
    
    def transform_with_scaler(self, standard_scaler: StandardScaler, input_data: pd.DataFrame) -> pd.DataFrame:
        """ Scales the inference data with the saved StandardScaler 
            :argument: scaler - Loaded Sklearn StandardScaler from the MLflow artifacts
            :argument: input_data - Pandas Dataframe containing inference data
            :return: scaled_data - Pandas Dataframe with scaled values
        """
        scaled_data = input_data.copy()
        self.logger.info("Scaling the dataset based on condition...")
        sensors = [e for e in list(scaled_data.columns) if 'sensor_' in e]
        for condition in scaled_data['condition'].unique():
            scaled_data.loc[scaled_data['condition'] == condition, sensors] = \
            standard_scaler.transform(scaled_data.loc[scaled_data['condition'] == condition, sensors])
        self.logger.info("Dataset successfully scaled!")
        return scaled_data
    
    def select_features(self, input_data: pd.DataFrame, run_id: str) -> pd.DataFrame:
        """ Selects the features from the saved MLflow artifacts that corresponds to the features 
            used to train the model
            :argument: input_data - Pandas Dataframe containing inference data
            :argument: run_id - Mlflow run Id for the loaded model
            :return: selected_df - Dataframe with data for only specified features/columns
        """
        self.logger.info("Selecting the features from input data...")
        os.makedirs("downloaded_artifacts", exist_ok=True)
        # Check if features JSON exists in MLflow
        try:
            features_path = self.mf_client.download_artifacts(run_id, "Features/features.json", "downloaded_artifacts")
        except:
            warning = f"There is no saved features JSON!"
            self.logger.warning(warning)
            raise FileNotFoundError(warning)
        else:
            # Get the features list from MLflow
            self.logger.info("Loading features from MLflow...")
            features_json = json.load(open(features_path))
            features_list = features_json['Features']
            # Remove the local file downlaoded by MLflow
            os.remove(features_path)
            # Select features
            selected_df = input_data[features_list]
        self.logger.info("Feature selection done!")
        return selected_df
    
    def predict(self, model: mlflow.pyfunc.PyFuncModel, input_data: pd.DataFrame) -> pd.DataFrame:
        """ Gets the prediction with selected model on given input_data 
            :argument: model - Loaded Mlflow model
            :argument: input_data - Pandas DataFrame containing scaled inference data
            :return: prediction - Numpy array with model predictions
        """
        self.logger.info(f"Predicting with the model: {self.model_name}")
        # Get predictions on input_data
        prediction = model.predict(input_data)
        # Return the prediction
        self.logger.info("Predictions finished!")
        return prediction
    
    def format_predictions(self, input_data: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
        """ Formats the predictions with the raw input data to a format ready for 
            Model Monitoring Athena table
            :argument: input_data - Pandas DataFrame containing raw inference data
            :argument: prediction - Numpy array with model predictions
            :return: formatted_data - Pandas DataFrame with concatenated input data and predictions
        """
        self.logger.info("Formatting input and output data...")
        formatted_data = input_data.copy()
        formatted_data['prediction'] = prediction
        # Add metadata of model used to the DataFrame
        formatted_data.insert(0, 'model_name', self.model_name)
        formatted_data.insert(1, 'version', self.model_version)
        formatted_data.insert(2, 'stage', self.stage)
        # Return the formatted data
        self.logger.info("Data successfully formated!")
        return formatted_data
        
    def save_to_parquet(self, s3_bucket: str, formatted_data: pd.DataFrame) -> None:
        """ Saves the concatenated data in the Glue table, ready to be used by Model 
            Monitoring pipeline, creates table if not existing for each unique model_name
            :argument: formatted_data - Pandas DataFrame with concatenated input data and predictions 
            :return: None
        """
        self.logger.info("Saving input and output data to parquet...")
        # Define save path into the S3 results path
        path = f"s3://{s3_bucket}/inference_results/{self.model_name}"
        database = "mlops-glue-database"
        table = self.model_name.replace("-", "_")
        # Save to parquet
        awswrangler.s3.to_parquet(formatted_data, path=path, dataset=True, mode='append', 
                                  database=database, table=table, partition_cols=['version'],
                                  filename_prefix=f"{self.model_name}-", boto3_session=boto3.Session(region_name='us-east-1'))
        self.logger.info(f"Input/Output data saved to parquet and Glue table: {table}")