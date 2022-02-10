import os
import mlflow

from data_handler import DataHandler
from ml_handler import MLHandler
from datetime import datetime


if __name__ == '__main__':
    # Instantiate DataHandler and MLHandler
    bucket_name = "mlops-storage-bucket"
    data_handler = DataHandler(s3_bucket=bucket_name)
    ml_handler = MLHandler(experiment_name="Predictive-Maintenance")
    # =======================================================LOADING DATA==================================================
    # Get training and testing data S3 paths
    train_path = os.environ.get('TrainDataPath', None)
    test_path = os.environ.get('TestDataPath', None)
    # If paths are not specified
    if not train_path:
        # Use current date's data in S3 bucket
        current_date = datetime.now().strftime('%Y-%m-%d')
        train_path = f"curated/total/parquet/train/{current_date}-train.parquet"
        test_path = f"curated/total/parquet/test/{current_date}-test.parquet"
    # Load both train and test data
    train_data = data_handler.load_data(train_path)
    test_data = data_handler.load_data(test_path)
    # Perform condition defining
    train_data = data_handler.define_conditions(train_data)
    test_data = data_handler.define_conditions(test_data)
    # Get the features from ENV variable
    selected_features = os.environ['features'].split(',')
    # ==========================================================TRAINING====================================================
    # Start the MLflow run
    current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    with mlflow.start_run(run_name=f"XGBRegressor-{current_time}", tags={'ImageTag': os.environ['ImageTag']}) as run:
        # Standardize data with StandardScaler
        std_train_data, std_test_data = ml_handler.standardize_with_scaler(train_data, test_data)
        # Smooth the data
        smoothed_train_data = data_handler.smooth_data(std_train_data, window=10, target_clip=150)
        smoothed_test_data = data_handler.smooth_data(std_test_data, window=10)
        # Split the data on training and test sets
        x_train, y_train, x_test, y_test = ml_handler.define_ml_dataset(smoothed_train_data, smoothed_test_data,
                                                                        selected_features, run)
        # Log the dataset path in MLflow
        dataset_reference = {'TrainData': bucket_name + '/' + train_path,
                             'TestData': bucket_name + '/' + test_path}
        ml_handler.mf_client.log_dict(run.info.run_id, dataset_reference, "Dataset/data_reference.json")
        # Get parameters values from ENV variables
        n_estimators = [int(e) for e in os.environ['n_estimators'].split(',')]
        max_depth = [int(e) for e in os.environ['max_depth'].split(',')]
        min_child_weight = [int(e) for e in os.environ['min_child_weight'].split(',')]
        reg_lambda = [int(e) for e in os.environ['reg_lambda'].split(',')]
        gamma = [int(e) for e in os.environ['gamma'].split(',')]
        
        parameters = {'n_estimators': n_estimators, 'max_depth': max_depth,
                      'min_child_weight': min_child_weight, 'reg_lambda': reg_lambda, 'gamma': gamma}
        # Train the model
        best_model, grid_results = ml_handler.train_xgboost(x_train, y_train, parameters, run, smoothed_train_data)
        # Evaluate the model
        train_metrics, test_metrics = ml_handler.evaluate_model(best_model, x_train, y_train, x_test, y_test, run)
