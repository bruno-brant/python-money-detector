from typing import Dict, Literal, Optional, Union

import torch

from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from money_counter.data import Target
from money_counter.models import PredictedTarget

Alignment = Union[Literal['top'], Literal['bottom'], Literal['center']]


def render_boxes(ax: Axes, box_: torch.Tensor, label: str, color: str, position: Alignment):
    box = box_.numpy().astype(int)

    # render the box on the image
    rect = patches.Rectangle([box[0], box[1]], box[2] - box[0],
                             box[3] - box[1], linewidth=1, edgecolor=color, facecolor='none')

    ax.add_patch(rect)

    # Write value on the box
    match position:
        case 'top':
            ax.text(box[0], box[1], label, color=color, fontsize=12)
        case 'bottom':
            ax.text(box[0], box[3], label, color=color, fontsize=12)
        case 'center':
            ax.text((box[0] + box[2]) / 2, (box[1] + box[3]) /
                    2, label, color=color, fontsize=12)


def render_image_and_boxes(image: torch.Tensor, label_map: Dict[int, str],
                           target: Optional[Target] = None, predicted: Optional[PredictedTarget] = None,
                           min_score: float = 0.5):

    image = image.cpu()

    if target is not None:
        target = {k: v.cpu() for k, v in target.items()
                  if k != 'image_id'}  # type: ignore

    if predicted:
        predicted = {k: v.cpu() for k, v in predicted.items()}  # type: ignore

    _, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(image.permute(1, 2, 0))

    if target is not None:
        for box, label in zip(target['boxes'], target['labels']):
            render_boxes(ax, box, label_map[int(label.item())], 'red', 'top')

    if predicted is not None:
        for box, label, score in zip(predicted['boxes'], predicted['labels'], predicted['scores']):
            if score > min_score:
                text = f"{label_map[int(label.item())]} ({score:.2f})"
                render_boxes(ax, box, text, 'yellow', 'bottom')
