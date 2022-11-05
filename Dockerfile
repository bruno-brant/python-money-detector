# DOCKER FILE FOR A PYTHON FLASK SERVER (src/server.py)
# VERSION 1.0.0
FROM python:3.10.0a6-alpine3.13

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD ./src /app
