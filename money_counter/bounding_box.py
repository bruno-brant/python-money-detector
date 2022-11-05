"""
Some bounding box operations.
"""
from typing import Tuple


def get_bbox_area(bbox: Tuple[int, int, int, int]) -> float:
    """
    Returns the area of a bounding box.
        :param bbox: A bounding box in the format (x1, y1, x2, y2).
        :returns: The area of the bounding box.
    """
    x1, y1, x2, y2 = bbox

    return (x2 - x1) * (y2 - y1)
