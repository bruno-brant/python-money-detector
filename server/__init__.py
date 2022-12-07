# create a small server to run the model and return the result
# import the necessary packages
import base64
import os
import io
from logging import getLogger
from logging.config import dictConfig
from typing import List, Optional, Tuple, TypedDict, cast

import flask
import torch
from flask import Flask, jsonify, request
from PIL import Image

from money_counter import models, prediction
from money_counter.utils import Timer

from money_counter.constants import CLASSES

# The directory where the model state is stored
# Models are expected to be like ./<MODEL_STATE_DIR>/<model_name>/epoch_<num>.pth
MODEL_STATE_DIR = os.environ.get('MODEL_STATE_DIR', './model_state')

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
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

logger = getLogger('server')


def initialize_predictor():
    model, model_name = models.get_fasterrcnn_untrained()
    version_manager = models.VersionManager(MODEL_STATE_DIR)
    device = torch.device('cpu')
    epoch, loss = version_manager.load_model(
        model_name, model, map_location=device)

    logger.info(f'Loaded model from epoch {epoch} with loss {loss}')

    return prediction.Predictor(model, model_name, device=device)


predictor = initialize_predictor()


app = Flask(__name__)


class BoundingBox(TypedDict):
    topLeft: Tuple[int, int]
    bottomRight: Tuple[int, int]


class CoinDescription(TypedDict):
    score: float
    label: str
    value: float
    boundingBox: BoundingBox


def _decode_label(label_idx: int) -> Tuple[str, float]:
    """
    :returns:
        A tuple of the label name and the value of the coin.
    """
    if not 0 <= label_idx < len(CLASSES):
        raise ValueError(f'Unknown label {label_idx}')

    return (CLASSES[label_idx], int(CLASSES[label_idx]) * 0.01 if CLASSES[label_idx].isdigit() else 0)


def _to_coin_description(boxes: torch.Tensor, label: int, score: float):
    label_txt, value = _decode_label(label)

    return CoinDescription(
        score=score,
        label=label_txt,
        value=value,
        boundingBox=BoundingBox(
            topLeft=(int(boxes[0]), int(boxes[1])),
            bottomRight=(int(boxes[2]), int(boxes[3]))
        ))


@app.after_request
def handle_cors(response: flask.Response) -> flask.Response:
    # if (request.method == 'OPTIONS'):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


@app.route("/api/v1/predict", methods=["POST"])
@app.route("/v1/predict", methods=["POST"])
def predict():
    logger.info(f'content type: {request.content_type}')

    if request.content_type == 'application/json':
        image_base64 = request.json['image']  # type: ignore
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))

    elif request.content_type.startswith('image/'):
        image = Image.open(io.BytesIO(request.data))  # type: ignore

    else:
        return jsonify({'error': 'unsupported content type'}), 400

    logger.info(f'image size: {image.size}')

    with Timer() as timer:
        logger.info('Predicting...')
        result = predictor.predict(image)
        logger.info('...finished.')

    result = {k: cast(torch.Tensor, v).cpu().tolist()
              for k, v in result.items()}

    # The coins that were detected
    coins = [_to_coin_description(boxes, label, score)
             for boxes, label, score in zip(result["boxes"], result["labels"], result["scores"])]

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
