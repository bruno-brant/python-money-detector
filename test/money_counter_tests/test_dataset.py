import PIL
import torch
import os

from money_counter import CoinsDataset

dir_path = os.path.dirname(os.path.realpath(__file__))

def test_coins_dataset():
    viafile_path = os.path.join(
        dir_path, '..\\..\\assets\\coins_dataset\\TCC_MBA_Coins.json')
    images_root_dir = os.path.join(dir_path, '..\\..\\assets\\coins_dataset\\images')

    ds = CoinsDataset(viafile_path, images_root_dir)

    image, target = ds[0]

    assert type(image) == PIL.Image.Image
    assert type(target) == dict
    assert type(target['boxes']) == torch.Tensor
    assert type(target['labels']) == torch.Tensor
    assert type(target['filename']) == str
