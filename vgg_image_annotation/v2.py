"""
Exports types that defines the content of the VGG Image Annotation (VIA) tool
save format.
"""
import json
from typing import Dict, List, Literal, Tuple, TypedDict, Union, cast

import numpy as np

# Define the type of the shape
ShapeAttributesCircle = TypedDict('ShapeAttributesCircle', {
    'name': Literal['circle'],
    'cx': int,
    'cy': int,
    'r': int
})

ShapeAttributesEllipse = TypedDict('ShapeAttributesEllipse', {
    'name': Literal['ellipse'],
    'cx': int,
    'cy': int,
    'rx': float,
    'ry': float,
    'theta': float
})

ShapeAttributesPolygon = TypedDict('ShapeAttributesPolygon', {
    'name': Literal['polygon'],
    'all_points_x': List[int],
    'all_points_y': List[int]
})

ShapeAttributesPolyline = TypedDict('ShapeAttributesPolyline', {
    'name': Literal['polyline'],
    'all_points_x': List[int],
    'all_points_y': List[int]
})

ShapeAttributesRect = TypedDict('ShapeAttributesRect', {
    'name': Literal['rect'],
    'x': int,
    'y': int,
    'width': int,
    'height': int
})


ShapeAttributes = Union[
    ShapeAttributesCircle,
    ShapeAttributesEllipse,
    ShapeAttributesPolygon,
    ShapeAttributesPolyline,
    ShapeAttributesRect
]

Edition = Union[Literal['New'], Literal['Old']]
Side = Union[Literal['Up'], Literal['Down']]

RegionAttributes = TypedDict('RegionAttributes', {
    'Value': str,
    'Edition': Edition,
    'Side': Side
})

Region = TypedDict('Region', {
    'shape_attributes': ShapeAttributes,
    'region_attributes': RegionAttributes
})

class ImageMetadata(TypedDict):
    """
    Contains metadata for a image, including annotated regions.
    """
    filename: str
    """The path to the image."""
    size: int
    """The size of the image in bytes."""
    regions: List[Region]
    """The regions in the image."""

class CoreSettings(TypedDict):
    buffer_size: int
    """The size of the buffer used to store the images."""
    filepath: str
    """The path to the file containing the images."""
    default_filepath: str
    """The default path to the file containing the images."""

class Settings(TypedDict):
    """
    Settings section of via file
    """
    ui: Dict[str, str]
    """The UI settings."""
    core: CoreSettings
    """The core settings."""
    project: Dict[str, str]
    """The project settings."""


ViaV2SaveFileFormat = TypedDict('ViaV2SaveFileFormat', {
    '_via_settings': Settings,
    '_via_img_metadata': Dict[str, ImageMetadata],
    '_via_data_format_version': str
})

Point = Tuple[int, int]
BoundingBox = Tuple[Point, Point]


def load_via_v2_file(filename: str) -> ViaV2SaveFileFormat:
    """
    Loads a VIA V2 save file.
    :param filename: The filename of the VIA V2 save file.
    :returns: The VIA V2 save file.
    """

    with open(filename, 'r') as f:
        return json.load(f)


def get_bounding_box(shape: ShapeAttributes, size: Tuple[int, int]) -> BoundingBox:
    """
    Returns the bounding box of the shape.
    :param shape: The shape to get the bounding box of.
    :param size: The size of the image, in (width, height) format.

    :returns:
        ((x1, y1), (x2, y2)), the top left and bottom right corners of
        the bounding box.
    """
    name = shape['name']

    if name == 'circle':
        shape = cast(ShapeAttributesCircle, shape)
        cx = shape['cx']
        cy = shape['cy']
        r = shape['r']

        topleft = (cx - r, cy - r)
        bottomright = (cx + r, cy + r)

    elif name == 'ellipse':
        shape = cast(ShapeAttributesEllipse, shape)
        cx = shape['cx']  # x-position of the center
        cy = shape['cy']  # y-position of the center
        rx = shape['rx']  # radius on the x-axis
        ry = shape['ry']  # radius on the y-axis
        # theta = shape['theta']

        # rotate the ellipse by theta degrees
        topleft = np.array([cx - rx, cy - ry])
        bottomright = np.array([cx + rx, cy + ry])

        # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # center = np.array([cx, cy])
        # rotate the bounding by theta
        # topleft = (topleft[0] * np.cos(theta) - topleft[1] * np.sin(theta),
        #            topleft[0] * np.sin(theta) + topleft[1] * np.cos(theta))
        # bottomright = (bottomright[0] * np.cos(theta) - bottomright[1] * np.sin(
        #     theta), bottomright[0] * np.sin(theta) + bottomright[1] * np.cos(theta))

    elif name == 'polyline':
        shape = cast(ShapeAttributesPolyline, shape)
        # Get the bounding box for the polyline
        topleft = np.array([min(shape['all_points_x']),
                           min(shape['all_points_y'])])
        bottomright = np.array(
            [max(shape['all_points_x']), max(shape['all_points_y'])])

    elif name == 'polygon':
        shape = cast(ShapeAttributesPolygon, shape)
        # Get the bounding box for the polygon
        topleft = np.array([min(shape['all_points_x']),
                           min(shape['all_points_y'])])
        bottomright = np.array(
            [max(shape['all_points_x']), max(shape['all_points_y'])])

    elif name == 'rect':
        shape = cast(ShapeAttributesRect, shape)
        topleft = (shape['x'], shape['y'])
        bottomright = (shape['x'] + shape['width'],
                       shape['y'] + shape['height'])

    else:
        raise ValueError(f'Unknown shape {name}')

    # Make sure the bounding box is within the image
    topleft = [max(c, 0) for c in topleft]
    bottomright = [min(c, s) for c, s in zip(bottomright, size)]

    return topleft, bottomright
