#FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
FROM cnstark/pytorch:1.13.0-py3.9.12-cuda11.7.1-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Create certificate files from env variable
COPY cert.pem /app/server.crt
COPY cert.key /app/server.key

ADD ./requirements.txt /app

# Upgrade pip
RUN pip install --upgrade pip

# Install packages
RUN pip3 install -r requirements.txt 
RUN pip3 install gunicorn

# Copy the source code
COPY server /app/server/
COPY vgg_image_annotation /app/vgg_image_annotation
COPY money_counter /app/money_counter

# Download the model state file
ADD https://moneycounter.blob.core.windows.net/models/model_state/fasterrcnn_resnet50_fpn-pretrained/fasterrcnn_resnet50_fpn-untrained/epoch_042.pth /app/model_state/fasterrcnn_resnet50_fpn-untrained/epoch_042.pth

# Make sure the HTTPS port is exposed
EXPOSE 443

COPY gunicorn.conf.py /app

# Run a WSGI server
CMD ["gunicorn",  "server:app"]
