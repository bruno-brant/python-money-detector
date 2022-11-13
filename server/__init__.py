# create a small server to run the model and return the result
# import the necessary packages
import base64
import io
import os
from typing import cast

import flask
import torch
from flask import Flask, jsonify, request
from PIL import Image

from money_counter import constants, models, prediction


def initialize_predictor():
    model, model_name = models.get_fasterrcnn_untrained()
    version_manager = models.VersionManager(constants.MODEL_STATE_DIR)
    epoch, loss = version_manager.load_model(model_name, model)
    
    print(f'Loaded model from epoch {epoch} with loss {loss}')

    return prediction.Predictor(model, model_name)

predictor = initialize_predictor()

app = Flask(__name__)

def decode_data_url(data_url: str) -> Image.Image:
    """Decode a data URL into a PIL image.

    Args:
        data_url (str): The data URL to decode.

    Returns:
        Image.Image: The decoded image.
    """
    # split the data URL into its components
    data_url = data_url.split(',')[1]

    # decode the image and return it
    image = Image.open(io.BytesIO(base64.b64decode(data_url)))
    return image


@app.after_request
def handle_cors(response: flask.Response) -> flask.Response:
    #if (request.method == 'OPTIONS'):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


@app.route("/predict", methods=["POST"])
def predict():
    print("predicting image...")

    if request.content_type == 'application/json':
        image_base64 = request.json['image']  # type: ignore
        #image = decode_data_url(image_data_url)
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))

    elif request.content_type ==  'image/jpeg' or 'image/png':
        image = Image.open(io.BytesIO(request.data))  # type: ignore

    else:
        return jsonify({'error': 'unsupported content type'}), 500
        
    print(f'image size: {image.size}')

    result = predictor.predict(image)
    result = {k: cast(torch.Tensor, v).cpu().numpy().tolist()
              for k, v in result.items()}

    return jsonify(result), 200

# Hello world flask


@app.route("/")
def hello():
    return "Hello World!"


# if __name__ == "__main__":
#     predictor = initialize_predictor()
#     app = initialize_flask(predictor)

#     # predict route that receives the image and calls predict
#     app.run(debug=True)
