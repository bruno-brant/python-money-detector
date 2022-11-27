from money_counter import models
from scripts.train_model import train_model

# Get the model for training
model, model_name = models.get_fasterrcnn_v2_pretrained()

if __name__ == '__main__':
    train_model(model, model_name)
