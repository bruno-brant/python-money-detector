import os

import torch

from money_counter import constants

print(os.getcwd())


def print_losses(folder: str):
    best = (float('inf'), None)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            if file.endswith('.pth'):
                model = torch.load(path)

                print(f'Loss for {path}:', model['loss'])

                if best is None or best[0] > model['loss']:
                    best = (model['loss'], path)

        elif os.path.isdir(path):
            print_losses(path)

    if best[1] is not None:
        print(f'Best model for {folder}:', best[1])


print_losses(constants.MODEL_STATE_DIR)
