import os
from typing import Dict, Literal, Optional, Tuple, TypedDict, Union

import torch
import torchvision
from torch import Tensor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import (FasterRCNN,
                                                      FastRCNNPredictor)

from money_counter.constants import NUM_CLASSES


class Target(TypedDict):
    """
    Description of a image target for the torchvision detection algorithms.
    """
    boxes: Tensor  # FloatTensor
    """The coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H"""

    labels: Tensor  # IntTensor
    """the label for each bounding box. 0 always represents the background class."""

    area: Tensor
    """The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes."""

    iscrowd: Tensor  # IntTensor
    """instances with iscrowd=True will be ignored during evaluation."""

    image_id: Tensor
    """Each image will have a unique id, which will be used during evaluation"""

class PredictedTarget(TypedDict):
    """
    Result of a prediction from the torchvision detection algorithms.
    """
    boxes: Tensor  # FloatTensor
    """The coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H"""
    scores: Tensor  # FloatTensor
    """The confidence score for each one of the N bounding boxes"""
    labels: Tensor  # IntTensor
    """the label for each bounding box. 0 always represents the background class."""


def get_fasterrcnn_pretrained() -> Tuple[FasterRCNN, str]:
    """
    Constructs a Faster R-CNN model with a pre-trained backbone.
    """
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES + 1)

    return model, "fasterrcnn_resnet50_fpn"


def get_fasterrcnn_untrained() -> Tuple[FasterRCNN, str]:
    """
    Constructs a Faster R-CNN model with an untrained backbone.
    """
    # load a model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES + 1)

    return model, "fasterrcnn_resnet50_fpn"


def get_fasterrcnn_v2_pretrained() -> Tuple[FasterRCNN, str]:
    """
    Constructs a Faster R-CNN model with an untrained backbone.
    """
    # load a model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES + 1)

    return model, "fasterrcnn_resnet50_fpn_v2"


def get_model(model_name) -> torch.nn.Module:
    """Gets the model for the given model name."""
    if model_name == 'fasterrcnn_resnet50_fpn':
        return get_fasterrcnn_pretrained()[0]

    if model_name == 'fasterrcnn_resnet50_fpn_pretrained':
        return get_fasterrcnn_pretrained()[0]

    raise ValueError(f'Unknown model name: {model_name}')


Checkpoint = TypedDict('Checkpoint', {
    'epoch': int,
    'model_state_dict': Dict,
    'optimizer_state_dict': Dict,
    'loss': float
})

Modes = Union[Literal["last"], Literal["best"]]


class VersionManager:
    """Used to save and load the model state to/from the model_state_dir"""

    def __init__(self, model_state_dir):
        """
        Initialize the VersionManager.
        :param model_state_dir: The directory where the model states are saved.
        """
        self._model_state_dir = model_state_dir

    def load_model(self, model_name: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, mode: Modes = "last", map_location: Optional[torch.device] = None) -> Tuple[int, float]:
        """Load the model from the model_state_dir"""
        model_path = self._get_model_path(model_name, mode=mode)

        if not os.path.exists(model_path):
            print(f'Could not find model at {model_path}')
            return 0, 0

        print(f'Loading model from {model_path}')

        checkpoint: Checkpoint = torch.load(
            model_path, map_location=map_location)

        # load the model mapping to cpu
        model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'], )

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f'Loaded model from {model_path}')

        return epoch, loss

    def save_model(self, model_name: str, model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer, epoch: int, loss: float):
        """Save the model to the model_state_dir"""
        model_path = self._get_model_path(model_name, epoch)

        print(f'Saving model to {model_path}')

        if os.path.exists(model_path):
            print(f'Backuping model file at epoch "{epoch}"...')

            if os.path.exists(f'{model_path}.bak'):
                os.remove(f'{model_path}.bak')

            os.rename(model_path, f'{model_path}.bak')

        # create the directory if it does not exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)

        print(f'Saved model to {model_path}')

    def _get_model_path(self, model_name, epoch: Optional[int] = None, mode: Modes = "last", map_location: Optional[torch.device] = None) -> str:
        """
        Format the path to the model file based on the model name and 
        the epoch.

        :param model_name: 
            The name of the model
        :param epoch: 
            The epoch of the model. Optional. If not provided,
            the latest model will be returned.
        """
        dir = f'{self._model_state_dir}/{model_name}'

        if epoch is not None:
            return f'{self._model_state_dir}/{model_name}/epoch_{epoch:03}.pth'

        if os.path.exists(dir):
            files = [file for file in os.listdir(dir) if file.endswith('.pth')]
            # , key=lambda x: int(x.split('_')[1].split('.')[0]))
            files = sorted(files)

            if mode == "last":
                latest = files[-1]
                return f'{dir}/{latest}'

            if mode == "best":
                for file in files:
                    file_list = [f'{dir}/{file}' for file in files]
                    losses = [(torch.load(file, map_location=map_location)['loss'], file)
                              for file in file_list]
                    file = min(zip(file_list, losses),
                               key=lambda x: x[1])[0]
                    return file

            return f'{self ._model_state_dir}/{model_name}/epoch_00.pth'

        return ""
