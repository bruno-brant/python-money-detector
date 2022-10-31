import os

import torch

from money_counter import constants

print(os.getcwd())


def copy_best(folder: str):
    best: dict = {'loss': float('inf')}

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path) and file.endswith('.pth'):
            model = torch.load(path)
            model['path'] = path

            print(f'Loss for {path}:', model['loss'])

            if best is None or best['loss'] > model['loss']:
                best = model

        elif os.path.isdir(path):
            copy_best(path)

    if best['loss'] != float('inf'):
        print(f'Best model for {folder}:', best['path'])
        os.makedirs(constants.MODEL_FINAL_DIR, exist_ok=True)
        torch.save(best, os.path.join(constants.MODEL_FINAL_DIR,
                   os.path.basename(folder) + '.pth'))


copy_best(constants.MODEL_STATE_DIR)
