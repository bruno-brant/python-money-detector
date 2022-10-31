import os
import torch

from money_counter import models, prediction, constants


def get_model():
    model, model_name = models.get_fasterrcnn_pretrained()
    state_path = os.path.join(constants.MODEL_FINAL_DIR, model_name + '.pth')
    state = torch.load(state_path)
    
    model.load_state_dict(state['model_state_dict'])
    
    return model, model_name


def test_predictor():
    model, model_name = get_model()
    predictor = prediction.Predictor(model, model_name)
    
    #predictor.predict()
