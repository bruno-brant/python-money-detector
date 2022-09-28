# Shared constants for all experiments
CLASSES = [5, 10, 25, 50, 100] 						# We DON'T have background class...
"""All the classes that are in the dataset"""

NUM_CLASSES = len(CLASSES) 
"""Number of classes in the dataset"""

BATCH_SIZE = 4 # Best for my GPU
NUM_EPOCHS = 10
ASSETS_DIR = '../assets'
DATASET_DIR = f'{ASSETS_DIR}/coins_dataset'
MODEL_STATE_DIR = f'../model_state'
