import argparse
import os
from money_counter import data, models, training

DATASET_DIR = f'../assets/coins_dataset'
MODEL_STATE_DIR = f'../model_state'

# Get the model for training
model, model_name = models.get_fasterrcnn_v2_pretrained()

# Create a parser with the DATASET directory and the MODEL_STATE directory as arguments
parser = argparse.ArgumentParser(
    description='Train a FasterRCNN model on the coins dataset')
parser.add_argument('--dataset-dir', type=str,
                    default=DATASET_DIR, dest='dataset_dir')
parser.add_argument('--model-state-dir', type=str,
                    default=MODEL_STATE_DIR, dest='model_state_dir')

if __name__ == '__main__':
    args = parser.parse_args()

    # Get the data
    data_loader_train, data_loader_test = data.get_data_loaders(
        dataset_path=os.path.join(args.dataset_dir, 'coins.json'))

    # Train the model
    training.train(model, model_name, data_loader_train, data_loader_test,
                   state_dir=args.model_state_dir, print_frequency=25, num_epochs=150)
