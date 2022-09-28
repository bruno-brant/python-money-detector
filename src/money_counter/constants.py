# Shared constants for all experiments
CLASSES = ['Background', 'Unknown', '5', '10', '25', '50', '100']
"""All the classes that are in the dataset"""

NUM_CLASSES = len(CLASSES) 
"""Number of classes in the dataset"""

BATCH_SIZE = 4
"""Batch size for training"""

ASSETS_DIR = '../assets'
DATASET_DIR = f'{ASSETS_DIR}/coins_dataset'
MODEL_STATE_DIR = f'./model_state'
