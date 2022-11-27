from train_model import train_model

from money_counter import models

# Get the model for training
model, model_name = models.get_fasterrcnn_v2_untrained()


if __name__ == '__main__':
    train_model(model, model_name)
