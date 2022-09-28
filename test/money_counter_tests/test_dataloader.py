import os
import PIL
import torch
from money_counter import data

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_dataloader():
    ds = data.CoinsDataset(
        f'{dir_path}\\..\\..\\assets\\coins_dataset\\TCC_MBA_Coins.json', f'{dir_path}\\..\\..\\assets\\coins_dataset\\images')
    loader = data.CoinsDataLoader(ds, 2, shuffle=False)

    images, targets = next(iter(loader))

    assert type(images) == list
    assert type(targets) == list
    assert type(images[0]) == PIL.Image.Image
    assert type(targets[0]) == dict
    assert type(targets[0]['boxes']) == torch.Tensor
    assert type(targets[0]['labels']) == torch.Tensor
    assert type(targets[0]['filename']) == str


def test_bboxes_inside_image():
    ds = data.CoinsDataset(
        f'{dir_path}\\..\\..\\assets\\coins_dataset\\TCC_MBA_Coins.json', f'{dir_path}\\..\\..\\assets\\coins_dataset\\images')
    loader = data.CoinsDataLoader(ds, 10, shuffle=False)

    for images, targets in loader:
        for image, target in zip(images, targets):
            for box in target['boxes']:
                top, left, bottom, right = box
                assert top >= 0, f'Box top is outside image: {box}'
                assert left >= 0, f'Box left is outside image: {box}'
                assert bottom <= image.width, f'Box bottom is outside image: {box}'
                assert right <= image.height, f'Box right is outside image: {box}'
