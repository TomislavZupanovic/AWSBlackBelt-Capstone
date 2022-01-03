FROM public.ecr.aws/bitnami/python:3.8-debian-10

# Copy Function code
COPY etl_lambda.py ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using requirements.txt file
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Set the CMD to a lambda_handler
CMD ["etl_lambda.lambda_handler"]