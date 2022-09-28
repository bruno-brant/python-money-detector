from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from money_counter.models import Target
from vgg_image_annotation import v2

from money_counter.bounding_box import get_bbox_area
from money_counter.utils import encode_data

def to_target(
        metadata: v2.ImageMetadata, image_size: Tuple[int, int], label_map: Dict[Any, int], filename_map: Dict[Any, int]) -> Target:
    """
    Converts a image metadata into a Target.
        :param metadata: 
                        The image metadata.
        :param label_map: 
                        A dictionary mapping label names to label indices. Will be updated with new labels.
        :param filename_map: 
                        A dictionary mapping filenames to image indices. Will be updated with new filenames.
        :returns: 
                        A Target dictionary.
    """
    regions = metadata['regions']
    values = [region['region_attributes']['Value'] for region in regions]
    bounding_boxes = _get_boxes_for_image(image_size, metadata)

    target: Target = {
        'boxes': torch.tensor(bounding_boxes, dtype=torch.float32),
        'labels': torch.tensor([*encode_data(label_map, values)], dtype=torch.int64),
        'image_id': torch.tensor([*encode_data(filename_map, [metadata['filename']])]),
        'area': torch.tensor([get_bbox_area(bbox) for bbox in bounding_boxes], dtype=torch.float32),
        'iscrowd': torch.tensor([1] * len(bounding_boxes), dtype=torch.int64)
    }

    return target


def is_annotated(image_metadata: v2.ImageMetadata) -> bool:
    """
    Check if the image has any annotated regions.
    """
    regions = image_metadata.get('regions', [])

    return any(filter(is_region_annotated, regions))


def is_region_annotated(region: v2.Region) -> bool:
    """
    Check if the region has any annotated attributes.
    """
    value = region['region_attributes'].get('Value', None)
    return value is not None


def _get_boxes_for_image(image_size: Tuple[int, int], image_metadata: v2.ImageMetadata) -> List[Tuple[int, int, int, int]]:
    list = []

    for region in image_metadata['regions']:
        shape = region['shape_attributes']
        box = v2.get_bounding_box(shape, image_size)
        topleft, bottomright = box

        list.append([*topleft, *bottomright])

    return list
