import argparse
from money_counter import data, models, training

DATASET_DIR = f'../assets/coins_dataset'
MODEL_STATE_DIR = f'../model_state'


# Create a parser with the DATASET directory and the MODEL_STATE directory as arguments
parser = argparse.ArgumentParser(
    description='Train a FasterRCNN model on the coins dataset')
parser.add_argument('--model-name', type=str, dest='model_name')
parser.add_argument('--dataset-path', type=str,
                    default=DATASET_DIR, dest='dataset_path')
parser.add_argument('--model-state-dir', type=str,
                    default=MODEL_STATE_DIR, dest='model_state_dir')

if __name__ == '__main__':
    args = parser.parse_args()

    # Get the model for training
    model = models.get_model(args.model_name)

    # create dataset from coco file
    CocoDetection()

    # Get the data
    data_loader_train, data_loader_test = data.get_data_loaders(args.dataset_path)

    # Train the model
    training.train(model, args.model_name, data_loader_train, data_loader_test,
                   state_dir=args.model_state_dir, print_frequency=20, num_epochs=100)
