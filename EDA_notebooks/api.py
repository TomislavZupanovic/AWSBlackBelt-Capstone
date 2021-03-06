import re
import requests
import os
import json

os.environ['API_URL'] = "https://as60shylek.execute-api.us-east-1.amazonaws.com/v1"

class MLOpsAPI:
    def __init__(self) -> None:
        pass

    def _format_parameters(self, payload: dict, parameters: dict) -> dict:
        """ Formats the sent parameters into the payload for REST API
            :argument: payload - Dictionary containing the mandatory parameters
            :argument: parameters - Dictionary containign arbitrary parameters
            :return: payload - Dictionary with both payload and parameters
        """
        for name, value in parameters.items():
            # Check if parameter value is list
            if type(value) == list:
                # Cast list values to the strings
                value = [str(e) for e in value]
                # Join all string values into one string
                value = ','.join(value)
            else:
                value = str(value)
            # Append name and value of the parameters to the payload dict
            payload[name] = value
        return payload

    def start_training(self, training_data_path: str, image_tag: str = None, **parameters) -> None:
        """ Starts the trainig with the specified image tag, if not specified, latest image will be used, 
            accepts arbitrary number of parameters
            :argument: training_data - Path to the dataset file stored in S3 to train model on
            :argument: image_tag - String specifying the Docker Image tag
            :argument: **parameters - Arbitrary named parameters with values to pass to the API
            :return: None
        """
        # Define the payload dictionary
        payload = {'TrainDataPath': training_data_path, 'TestDataPath': training_data_path.replace('train', 'test')}
        if image_tag:
            payload['ImageTag'] = image_tag
        # Check if parameters are specified
        if parameters:
            payload = self._format_parameters(payload, parameters)
        # Send POST request to the API endpoint
        response = requests.post(url=os.environ['API_URL'] + '/start_training', data=json.dumps(payload))
        # Convert response to the JSON/Dictionary
        json_response = response.json()
        # Print the json response
        print(json.dumps(json_response, indent=4))
        
    def define_train_schedule(self, cron_expression: str, action: str = 'create', image_tag: str = None, **parameters) -> None:
        """ Creates/updates or deletes the Cron schedule for training depending on the action specified 
            :argument: cron_expression - Crons expression for defining timely schedule
            :argument: action - 'create' or 'delete' the schedule Rule
            :argument: image_tag - String specifying the Docker Image tag
            :argument: **parameters - Arbitrary named parameters with values to pass to the API
            :return: None
        """
        # Check action argument value
        if action != "create" and action != "delete":
            print("Warning: action must be 'create' or 'delete'! (Default: create)")
            return None
        # Define the payload dictionary
        payload = {'Cron': cron_expression, 'Action': action}
        if image_tag:
            payload['ImageTag'] = image_tag
        # Check if parameters are specified
        if parameters:
            payload = self._format_parameters(payload, parameters)
        else:
            payload['Status'] = "NoAdditionalParameters"
        # Send POST request to the API endpoint
        response = requests.post(url=os.environ['API_URL'] + '/training_schedule', data=json.dumps(payload))
        # Convert response to the JSON/Dictionary
        json_response = response.json()
        # Print the json response
        print(json.dumps(json_response, indent=4)) 

    def start_batch_inference(self, model_name: str, inference_data: str, stage: str, image_tag: str = None, **parameters) -> None:
        """ Start the Batch Inference for the specified model name and data file name from LakeFS 
            :argument: model_name - Name of the MLflow model to use as predictor
            :argument: inference_data - Path of the dataset file stored in S3 to get predictions from
            :argument: stage - Name of the MLflow model stage
            :argument: image_tag - String specifying the Docker Image tag
            :argument: **parameters - Arbitrary named parameters with values to pass to the API
            :return: None
        """
         # Define the payload dictionary
        payload = {'ModelName': model_name, 'InferenceDataPath': inference_data, 'Stage': stage}
        if image_tag:
            payload['ImageTag'] = image_tag
        # Check if parameters are specified
        if parameters:
            payload = self._format_parameters(payload, parameters)
        # Send POST request to the API endpoint
        response = requests.post(url=os.environ['API_URL'] + '/start_batch_inference', data=json.dumps(payload))
        # Convert response to the JSON/Dictionary
        json_response = response.json()
        # Print the json response
        print(json.dumps(json_response, indent=4))
        
    def define_inference_schedule(self, cron_expression: str, model_name: str, stage: str, action: str = 'create',
                                  image_tag: str = None, **parameters) -> None:
        """ Creates/updates or deletes the Cron schedule for inference depending on the action specified 
            :argument: cron_expression - Crons expression for defining timely schedule
            :argument: model_name - Name of the MLflow model to use as predictor
            :argument: stage - Name of the MLflow model stage
            :argument: action - 'create' or 'delete' the schedule Rule
            :argument: image_tag - String specifying the Docker Image tag
            :argument: **parameters - Arbitrary named parameters with values to pass to the API
            :return: None
        """
         # Define the payload dictionary
        payload = {'ModelName': model_name, 'Cron': cron_expression, 'Action': action, 'Stage': stage}
        if image_tag:
            payload['ImageTag'] = image_tag
        # Check if parameters are specified
        if parameters:
            payload = self._format_parameters(payload, parameters)
        else:
            payload['Status'] = "NoAdditionalParameters"
        # Send POST request to the API endpoint
        response = requests.post(url=os.environ['API_URL'] + '/inference_schedule', data=json.dumps(payload))
        # Convert response to the JSON/Dictionary
        json_response = response.json()
        # Print the json response
        print(json.dumps(json_response, indent=4))
    