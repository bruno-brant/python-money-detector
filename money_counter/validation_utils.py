from typing import SupportsIndex, overload
import typing
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor, ToPILImage
from typing import Iterable, List, Literal, Optional, Tuple, TypeGuard, Union, cast

import torch

from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from money_counter.data import Target
from money_counter.models import PredictedTarget

Alignment = Union[Literal['top'], Literal['bottom'], Literal['center']]

TensorDict = List[Tuple[str, torch.Tensor]]


class LabelGetter(typing.Protocol):
    """Interface for getting the label corresponding to a number"""

    def __getitem__(self, __i: SupportsIndex) -> str:
        """Gets the corresponding label for the given index"""
        ...


def _draw_bounding_box(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, outline: str, fill: str):
    # The rect that is the bounding box
    draw.rectangle(tuple(box), outline=outline, fill=fill)

    x0, y0, x1, y1 = box

    # Make font proportional to box size
    font_size = int(((x1 - x0) / 5) * 2.5)
    font = ImageFont.truetype("arial.ttf", font_size)

    font_bbox = font.getbbox(text)

    mid_x = ((x0 + x1) / 2)
    mid_y = ((y0 + y1) / 2)

    # Center the text in the box
    text_x = mid_x - (font_bbox[2] / 2)
    text_y = mid_y - (font_bbox[3] / 2)


    draw.text((text_x, text_y), text, fill=outline, font=font)


def _is_predicted(targetOrPredicted: Target | PredictedTarget) -> TypeGuard[PredictedTarget]:
    return ('scores' in targetOrPredicted)


Box = Tuple[int, int, int, int]
Label = int
Score = int


@overload
def _to_cpu(targetOrPredicted: Target) -> Iterable[Tuple[Box, Label]]:
    ...


@overload
def _to_cpu(targetOrPredicted: PredictedTarget) -> Iterable[Tuple[Box, Label, Score]]:
    ...


def _to_cpu(targetOrPredicted: Target | PredictedTarget) -> Iterable[Tuple[Box, Label, Score]] | Iterable[Tuple[Box, Label]]:
    boxes = targetOrPredicted['boxes'].cpu().numpy().astype(int)
    labels = targetOrPredicted['labels'].cpu().numpy().astype(int)

    if _is_predicted(targetOrPredicted):
        scores = targetOrPredicted['scores'].cpu().numpy().astype(float)

        return zip(boxes, labels, scores)

    return zip(boxes, labels)


@overload
def show_prediction(image: torch.Tensor, label_map: LabelGetter,
                    target: Optional[Target] = None, predicted: Optional[PredictedTarget] = None,
                    min_score: float = 0.5) -> Image.Image:
    ...


@overload
def show_prediction(image: Image.Image, label_map: LabelGetter,
                    target: Optional[Target] = None, predicted: Optional[PredictedTarget] = None,
                    min_score: float = 0.5) -> Image.Image:
    ...


def show_prediction(image: torch.Tensor | Image.Image, label_map: LabelGetter,
                    target: Optional[Target] = None, predicted: Optional[PredictedTarget] = None,
                    min_score: float = 0.5) -> Image.Image:

    if isinstance(image, torch.Tensor):
        image = cast(Image.Image, ToPILImage()(image.cpu()))
    else:
        image = image.copy()

    image = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))

    draw = ImageDraw.Draw(overlay)

    if target is not None:
        for box, label in _to_cpu(target):
            _draw_bounding_box(
                draw, box, label_map[label], 'green', '#00FF0030')

    if predicted:
        for box, label, score in _to_cpu(predicted):
            if score < min_score:
                continue
            _draw_bounding_box(draw, box, label_map[label], 'red', '#FF000030')

    image = Image.alpha_composite(image, overlay)

    return image
