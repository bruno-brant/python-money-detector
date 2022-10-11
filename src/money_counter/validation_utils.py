from typing import Dict, Literal, Optional, Tuple, Union

from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

from money_counter.data import Target

Alignment = Union[Literal['top'], Literal['bottom'], Literal['center']]

def render_boxes(ax: Axes, box: Tuple[int, int, int, int], label: str, color: str, position: Alignment):
    # render the box on the image
    rect = patches.Rectangle([box[0], box[1]], box[2] - box[0],
                             box[3] - box[1], linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    # Write value on the box
    match position:
        case 'top':
            ax.text(box[0], box[1], label, color=color, fontsize=16)
        case 'bottom':
            ax.text(box[0], box[3], label, color=color, fontsize=16)
        case 'center':
            ax.text((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, label, color=color, fontsize=16)


def render_image_and_boxes(image: Image.Image, label_map: Dict[int, str], 
    target: Optional[Target] = None, predicted: Optional[Target] = None, 
    min_score: float = 0.5):

    _, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(image.permute(1, 2, 0))

    if target is not None:
        for box, label in zip(target['boxes'], target['labels']):
            render_boxes(ax, box, label_map[label.item()], 'red', 'top')

    if predicted is not None:
        for box, label, score in zip(predicted['boxes'], predicted['labels'], predicted['scores']):
            if score > min_score:
                render_boxes(ax, box, label_map[label.item()], 'yellow', 'center')
