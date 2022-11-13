#FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
FROM cnstark/pytorch:1.13.0-py3.9.12-cuda11.7.1-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

ADD ./requirements.txt /app

# Upgrade pip
RUN pip install --upgrade pip

# Install packages
RUN pip3 install -r requirements.txt 
RUN pip3 install gunicorn

# If the above doesn't work, try this
COPY server /app/server/
COPY vgg_image_annotation /app/vgg_image_annotation
COPY money_counter /app/money_counter
COPY model_final /app/model_final

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run a WSGI server
#CMD ["gunicorn", "-b", "server:app", "-w", "4", "-k", "gevent", "--timeout", "120", "--log-level", "debug", "--log-file", "-", "server:app"]
CMD ["gunicorn", "-w", "4", "--log-level", "debug", "--log-file", "-", "server:app"]
