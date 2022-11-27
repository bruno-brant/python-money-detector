import argparse
import os
from logging import getLogger, basicConfig, INFO

from money_counter import data, training

logging = getLogger(__name__)

# Configure logging
basicConfig(level=INFO)

COINS_DATASET_PATH = os.environ['COINS_DATASET_PATH']
MODEL_STATE_DIR = os.environ.get('MODEL_STATE_DIR', './model_state')

# Create a parser with the DATASET directory and the MODEL_STATE directory as arguments
parser = argparse.ArgumentParser(
    description='Train a FasterRCNN model on the coins dataset')

parser.add_argument('--dataset-path', type=str,
                    default=COINS_DATASET_PATH, dest='dataset_path')
parser.add_argument('--model-state-dir', type=str,
                    default=MODEL_STATE_DIR, dest='model_state_dir')


def train_model(model, model_name):
    """
    Train a model on the coins dataset.
    :param model: The model to train.
    :param model_name: The name of the model.
    """
    logging.info('Training model: %s', model_name)

    args = parser.parse_args()

    # Get the data
    data_loader_train, data_loader_test = data.get_data_loaders(
        args.dataset_path)

    # Train the model
    training.train(model, model_name, data_loader_train, data_loader_test,
                   state_dir=args.model_state_dir, print_frequency=25, num_epochs=100)
