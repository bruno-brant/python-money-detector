# create a small server to run the model and return the result
# import the necessary packages
import base64
import io
import os
from typing import cast

import torch
from flask import Flask, jsonify, request
from PIL import Image

from money_counter import constants, models, prediction, utils


def initialize_predictor():
    model, model_name = models.get_fasterrcnn_pretrained()

    # load the model state
    state_path = os.path.join(constants.MODEL_FINAL_DIR, model_name + '.pth')
    state = torch.load(state_path)

    model.load_state_dict(state['model_state_dict'])

    predictor = prediction.Predictor(model, model_name)

    return predictor


predictor = initialize_predictor()


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    match request.content_type:
        case 'application/json':
            image_base64 = request.json['image']  # type: ignore
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))

        case 'image/jpeg' | 'image/png':
            image = Image.open(io.BytesIO(request.data))  # type: ignore
        case _:
            return jsonify({'error': 'unsupported content type'})

    result = predictor.predict(image)
    result = {k: cast(torch.Tensor, v).cpu().numpy().tolist() for k, v in result.items()}

    return result

# Hello world flask


@app.route("/")
def hello():
    return "Hello World!"


# if __name__ == "__main__":
#     predictor = initialize_predictor()
#     app = initialize_flask(predictor)

#     # predict route that receives the image and calls predict
#     app.run(debug=True)
