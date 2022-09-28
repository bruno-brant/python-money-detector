from typing import Dict, Literal, Union, Tuple

from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

from money_counter.data import Target


def render_boxes(ax: Axes, box: Tuple[int, int, int, int], label: str, color: str, position: Union[Literal['top'], Literal['bottom']]):
    # render the box on the image
    rect = patches.Rectangle([box[0], box[1]], box[2] - box[0],
                             box[3] - box[1], linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    # Write value on the box
    if position == 'top':
        ax.text(box[0], box[1], label, color=color, fontsize=16)
    elif position == 'bottom':
        ax.text(box[0], box[3], label, color=color, fontsize=16)


def render_image_and_boxes(image: Image.Image, target: Target, predicted: Target, label_map: Dict[int, str], min_score: float = 0.5):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(image.permute(1, 2, 0))

    # for target, color, position in zip([target, predicted], ['red', 'yellow'], ['top', 'bottom']):
    for box, label in zip(target['boxes'], target['labels']):
        render_boxes(ax, box, label_map[label.item()], 'red', 'top')

    for box, label, score in zip(predicted['boxes'], predicted['labels'], predicted['scores']):
        if score > min_score:
            render_boxes(ax, box, label_map[label.item()], 'yellow', 'bottom')
