"""
Contains tools for predicting the number of coins in an image.
"""
from typing import List, Optional, cast

import torch
from PIL import Image
from torchvision import transforms

from money_counter import models, utils, engine


class Predictor:
    """Predicts the total value of coins in an image."""

    def __init__(self, model: torch.nn.Module, model_name: str, *, device: Optional[torch.device] = None):
        """Initializes the predictor."""
        if device is None:
            self.device = utils.get_device()

        self._model = model.to(self.device).eval()

        self._transform = transforms.Compose(
            [transforms.ToTensor()])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> models.PredictedTarget:
        """
        Predicts the number of coins in an image.
        """
        image_t = cast(torch.Tensor, self._transform(image))
        image_t = image_t.to(self.device)

        result: List[models.PredictedTarget] = self._model([image_t])

        # Apply nms
        engine.apply_nms(result, 0.5)

        # Convert to numpy array
        return result[0]
