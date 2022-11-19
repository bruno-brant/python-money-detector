# create a small server to run the model and return the result
# import the necessary packages
import base64
import io
from logging.config import dictConfig
from typing import TypedDict, cast

import flask
import torch
from flask import Flask, jsonify, request
from PIL import Image

from money_counter import constants, models, prediction
from money_counter.utils import Timer

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['wsgi']
    }
})


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


def format_result(result: models.PredictedTarget, model_name):
    """Formats the result of a prediction.

    Args:
        result (models.PredictedTarget): The result to format.

    Returns:
        str: The formatted result.
    """


@app.after_request
def handle_cors(response: flask.Response) -> flask.Response:
    # if (request.method == 'OPTIONS'):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


@app.route("/api/v1/predict", methods=["POST"])
def predict():
    if request.content_type == 'application/json':
        image_base64 = request.json['image']  # type: ignore
        #image = decode_data_url(image_data_url)
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))

    elif request.content_type.startswith('image/'):
        image = Image.open(io.BytesIO(request.data))  # type: ignore

    else:
        return jsonify({'error': 'unsupported content type'}), 400

    print(f'image size: {image.size}')

    with Timer() as timer:
        result = predictor.predict(image)

    result = cast(models.PredictedTarget, {k: cast(torch.Tensor, v).cpu().numpy().tolist()
                                        for k, v in result.items()})

    # The coins that were detected
    coins = [{
        "score": float(score),
        "label": constants.CLASSES[int(label)],
        "value": int(constants.CLASSES[int(label)]) if constants.CLASSES[int(label)] else 0,
        "boundingBox": {
            "topLeft": [int(boxes[0]), int(boxes[1])],
            "bottomRight": [int(boxes[2]), int(boxes[3])]
        }
    } for boxes, label, score in zip(result["boxes"], result["labels"], result["scores"])]

    formatted = {
        "data": {
            "model": predictor.model_name,
            "processingTime": timer.elapsed(),
            "image": {
                "size": {
                    "width": image.size[0],
                    "height": image.size[1]
                },
                "format": image.format,
                "mode": image.mode
            },
            "coins": coins
        },
    }

    return jsonify(formatted), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({'status': 'ok'}), 200
