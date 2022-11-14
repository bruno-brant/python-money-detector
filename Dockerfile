#FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
FROM cnstark/pytorch:1.13.0-py3.9.12-cuda11.7.1-ubuntu20.04

# The SSL Certificate for serving the files
ARG SERVER_CERTIFICATE
ARG SERVER_KEY

# Set the working directory to /app
WORKDIR /app

ADD ./requirements.txt /app

# Upgrade pip
RUN pip install --upgrade pip

# Install packages
RUN pip3 install -r requirements.txt 
RUN pip3 install gunicorn

# Create certificate files from env variable
RUN echo $SERVER_CERTIFICATE > /app/server.crt
RUN echo $SERVER_KEY > /app/server.key

# Copy the source code
COPY server /app/server/
COPY vgg_image_annotation /app/vgg_image_annotation
COPY money_counter /app/money_counter

# Download the model state file
ADD https://moneycounter.blob.core.windows.net/models/model_state/fasterrcnn_resnet50_fpn-pretrained/epoch_028.pth /app/model_state/fasterrcnn_resnet50_fpn/epoch_028.pth

# Make port 8000 available to the world outside this container
EXPOSE 443

# Run a WSGI server
CMD ["gunicorn", "-w", "4", "--log-level", "debug", "-b", "0.0.0.0:443", "--log-file", "-", "--certfile", "/app/server.crt", "--keyfile", "/app/server.key", "server:app"]
