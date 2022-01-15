FROM public.ecr.aws/bitnami/python:3.8-debian-10

# Copy the code
COPY /inference /inference
COPY /training /training

# Install dependencies using requirements.txt file
COPY requirements.txt .
RUN pip install -r requirements.txt