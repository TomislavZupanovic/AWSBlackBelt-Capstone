import os
from datetime import datetime

from data_handler import DataHandler
from inference_handler import InferenceHandler

if __name__ == '__main__':
    # Instantiate DataHandler and InferenceHanlder
    bucket_name = "mlops-storage-bucket"
    data_handler = DataHandler(s3_bucket=bucket_name)
    inference_handler = InferenceHandler(model_name=os.environ['ModelName'], stage=os.environ['Stage'])
    # Get data path from ENV variable
    inference_data_path = os.environ.get('InferenceDataPath', None)
    # If paths are not specified
    if not inference_data_path:
        # Use current date's data in S3 bucket
        current_date = datetime.now().strftime('%Y-%m-%d')
        inference_data_path = f"curated/partitioned/parquet/inference/{current_date}-inference.parquet"
    # Load the data from S3
    inference_data = data_handler.load_data(inference_data_path)
    conditioned_data = data_handler.define_conditions(inference_data)
    # Get the trained model from MLflow
    model = inference_handler.get_model()
    # Get the saved StandardScaler object
    standard_scaler = inference_handler.load_scaler(run_id=model.metadata.run_id)
    # Transform the inference data with StandardScaler
    scaled_data = inference_handler.transform_with_scaler(standard_scaler=standard_scaler, input_data=conditioned_data)
    # Perform Moving Average smoothing
    smoothed_data = data_handler.smooth_data(input_data=scaled_data, window=10)
    # Perform Feature Selection
    model_input_data = inference_handler.select_features(input_data=smoothed_data, run_id=model.metadata.run_id)
    # Get predictions
    prediction = inference_handler.predict(model=model, input_data=model_input_data)
    # Format the predictions with raw input data
    formatted_data = inference_handler.format_predictions(input_data=inference_data, prediction=prediction)
    # Save data to Glue Table for Model Monitoring
    inference_handler.save_to_parquet(s3_bucket=bucket_name, formatted_data=formatted_data)