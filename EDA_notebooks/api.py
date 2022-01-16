from email.mime import image
import re
import requests
import os
import json

# TODO: Add API Urls
os.environ['API_URL'] = ""


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

    def start_training(self, training_data: str, image_tag: str = None, **parameters) -> None:
        """ Starts the trainig with the specified image tag, if not specified, latest image will be used, 
            accepts arbitrary number of parameters
            :argument: training_data - Name of the dataset file stored in LakeFS to train model on
            :argument: image_tag - String specifying the Docker Image tag
            :argument: **parameters - Arbitrary named parameters with values to pass to the API
            :return: None
        """
        # Define the payload dictionary
        payload = {'TrainingDataFileName': training_data}
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

    def start_batch_inference(self, model_name: str, inference_data: str, image_tag: str = None, **parameters) -> None:
        """ Start the Batch Inference for the specified model name and data file name from LakeFS 
            :argument: model_name - Name of the MLflow model to use as predictor
            :argument: inference_data - Name of the dataset file stored in LakeFS to get predictions from
            :argument: image_tag - String specifying the Docker Image tag
            :argument: **parameters - Arbitrary named parameters with values to pass to the API
            :return: None
        """
         # Define the payload dictionary
        payload = {'ModelName': model_name, 'InferenceDataFileName': inference_data}
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
        
    def define_inference_schedule(self, cron_expression: str, model_name: str, action: str = 'create',
                                  image_tag: str = None, **parameters) -> None:
        """ Creates/updates or deletes the Cron schedule for inference depending on the action specified 
            :argument: cron_expression - Crons expression for defining timely schedule
            :argument: model_name - Name of the MLflow model to use as predictor
            :argument: action - 'create' or 'delete' the schedule Rule
            :argument: image_tag - String specifying the Docker Image tag
            :argument: **parameters - Arbitrary named parameters with values to pass to the API
            :return: None
        """
         # Define the payload dictionary
        payload = {'ModelName': model_name, 'Cron': cron_expression, 'Action': action}
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
    
    
        
    