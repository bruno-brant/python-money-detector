import json
import os
from logging import getLogger
from typing import Callable, Dict, Generic, Iterable, Iterator, List
from typing import Optional as Opt
from typing import Tuple, TypeVar, cast

import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from money_counter.constants import CLASSES
from money_counter.models import Target
from money_counter.via_utils import is_region_annotated, to_target
from vgg_image_annotation import v2

DatasetItem = Tuple[torch.Tensor, Target]
"""Item returned by a Torch Dataset instance."""

ViaTransform = Callable[[Image.Image, Target], DatasetItem]
"""Callback to transform a target."""

_label_map = {label: i for i, label in enumerate(CLASSES)}

logging = getLogger(__name__)


class ViaDataset(Dataset[DatasetItem]):
    """
    The dataset of coins.

    Items are tuples of (image, target), where:
    - image is a PIL image of shape (H, W, C) in the range [0, 255]
    - target is a dictionary containing:
        - boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
        - labels (Int64Tensor[N]): the label for each bounding box
    """

    def __init__(
            self, annotations_path: str, *,
            transform: Opt[ViaTransform] = None):
        """
        Initialises a new instance of CoinsDataset.
        :param annotations_path: path to the json file produced by the VIA (VGG Image Annotation) tool.
        :param root_dir: directory with all the images.
        :param transform: optional transform to be applied on a sample.
        :param target_transform: optional transform to be applied on the target.
        """
        if not os.path.isfile(annotations_path):
            raise FileNotFoundError(f"File '{annotations_path}' not found.")

        self._annotations_path = annotations_path
        """Path to the json file produced by the VIA (VGG Image Annotation) tool."""

        self._transform = transform
        """Additional transform to be applied on the target and image."""

        self.filename_map: Dict[str, int] = {}
        """The map between the filenames and the encoded filenames."""

        with open(annotations_path) as f:
            via: v2.ViaV2SaveFileFormat = json.load(f)

        # Images are relative to the path of the annotations file
        annotations_file_dir = os.path.dirname(annotations_path)
        # And the are in a subfolder thats the default filepath
        default_filepath = via['_via_settings']['core']['default_filepath']
        # So we can get the root dir by joining the two
        self._root_dir = os.path.join(annotations_file_dir, default_filepath)
        """Directory with the images."""

        # Get the metadata fe images
        self._images_metadata = list(via['_via_img_metadata'].values())
        """List of images metadata"""

        # Cache the name of the images for faster access
        self._images_names = [metadata['filename']
                              for metadata in self._images_metadata]

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self._images_names)

    def __getitem__(self, idx: int):
        """Return the image and its metadata at the given index."""
        filename = self._images_names[idx]

        image = self._get_image(filename)
        metadata = self._images_metadata[idx]
        target = to_target(metadata, image.size, _label_map, self.filename_map)

        if self._transform:  # apply the transform
            image, target = self._transform(image, target)

        return image, target

    def _get_image(self, filename: str) -> Image.Image:
        """Get the image with the given name."""
        image_path = os.path.join(self._root_dir, filename)
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')

        return image


class ViaDatasetOnlyAnnotated(ViaDataset):
    """
    The dataset of coins, but only the annotated images.
    """

    def __init__(
            self, annotations_path: str, *, transform: Opt[ViaTransform] = None):
        """
        Initialises a new instance of CoinsDataset.
        :param viajson_path: path to the json file produced by the VIA (VGG Image Annotation) tool
        :param root_dir: directory with all the images.
        :param transform: optional transform to be applied on a sample.
        """
        super().__init__(annotations_path, transform=transform)

        # Remove the images that don't have any annotations
        self._images_metadata = [
            *self._get_annotated_images(self._images_metadata)]

        # Update the image name cache
        self._images_names = [metadata['filename']
                              for metadata in self._images_metadata]

    def _get_annotated_images(self, images_metadata: List[v2.ImageMetadata]) -> Iterator[v2.ImageMetadata]:
        """
        Filter out images that have at least one annotated region.
        """
        for image_metadata in images_metadata:
            regions = image_metadata.get('regions', [])

            # Get only regions that have annotations
            regions = [region for region in regions
                       if is_region_annotated(region)]

            # Skip images that don't have any annotated regions
            if len(regions) == 0:
                continue

            # Create a new dict with all in image_metadata except regions, and override regions with the filtered regions
            metadata = {
                k: v for k, v in image_metadata.items() if k != 'regions'
            } | {
                'regions': regions
            }

            yield cast(v2.ImageMetadata, metadata)


def collate_into_tensors(items: List[DatasetItem]):
    """
    Stacks multiple dataset items for more effective processing.

    Called by DataLoader, this method is responsible for producing a single DatasetItem 
    for a list of them, by collating those into a single instance.
    """
    # TODO: Why do we need this cast? Is zip not able to infer the type?
    images, targets = cast(
        Tuple[Iterable[Image.Image], Iterable[Target]], zip(*items))

    # Stack the images
    images = torch.stack(images)   # type: ignore

    # Stack the targets, padding the boxes and labels
    boxes = [target['boxes'] for target in targets]
    boxes = pad_sequence(boxes, batch_first=True, padding_value=0)

    labels = [target['labels'] for target in targets]
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    targets = {'boxes': boxes, 'labels': labels}

    return images, targets


def collate_into_lists(items: List[DatasetItem]) -> Tuple[List[Image.Image], List[Target]]:
    """
    Collates the list of tuples into a tuple of lists.
    """
    images = []
    targets = []

    for image, target in items:
        images.append(image)
        targets.append(target)

    return images, targets


TDL = TypeVar("TDL")

T1 = TypeVar('T1')
T2 = TypeVar('T2')


def to_via_transform(image_transform: Callable[[T1], T2]) -> Callable[[T1, Target], Tuple[T2, Target]]:
    """
    Wraps a transform that only transforms the image into a transform that transforms both the image and the target.
    """
    def _transform(image: T1, target: Target) -> Tuple[T2, Target]:
        transformed_image = image_transform(image)
        return transformed_image, target

    return _transform


TLast = TypeVar('TLast')


class ComposeViaTransform(Generic[TLast]):
    """
    Composes multiple transforms into a single one.
    :note: 
        Similar to torchvision.transforms.Compose, but for ViaTransforms.
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self._transforms = transforms

    def __call__(self, image: Image.Image, target: Target) -> Tuple[TLast, Target]:
        for transform in self._transforms:
            image, target = transform(image, target)

        return cast(TLast, image), target


class NormalizeImageSize:
    """
    Normalize an image to a fixed size.

    Some images where captured with a small difference in size. This transform
    normalizes the images to a fixed size, by transforming the image to either
    4000x3000 or 3000x4000, depending on the aspect ratio.
    """

    def __init__(self, width: int = 4000, height: int = 3000):
        self._height = height
        self._width = width

    def __call__(self, image: Image.Image, target: Target):
        size = image.size
        if size[0] > size[1]:
            image = image.resize((self._width, self._height))
            target = self._resize_target(
                target, size, (self._width, self._height))
        else:
            image = image.resize((self._height, self._width))
            target = self._resize_target(
                target, size, (self._height, self._width))

        return image, target

    def _resize_target(self, target: Target, old_size: Tuple[int, int], new_size: Tuple[int, int]) -> Target:
        """
        Resizes a target to a new size.
        """
        boxes = target['boxes']
        boxes[:, 0] = boxes[:, 0] / old_size[0] * new_size[0]
        boxes[:, 1] = boxes[:, 1] / old_size[1] * new_size[1]
        boxes[:, 2] = boxes[:, 2] / old_size[0] * new_size[0]
        boxes[:, 3] = boxes[:, 3] / old_size[1] * new_size[1]
        target['boxes'] = boxes
        return target


default_transform = ComposeViaTransform[torch.Tensor]([
    NormalizeImageSize(),
    to_via_transform(transforms.ToTensor()),
    to_via_transform(transforms.ConvertImageDtype(torch.float)),
])


def get_data_loaders(
        dataset_path: str,
        *,
        train_transform: Opt[ViaTransform] = default_transform,
        test_transform:  Opt[ViaTransform] = default_transform,
        batch_size=3,
        test_percentage=0.2) -> Tuple[DataLoader[DatasetItem], DataLoader[DatasetItem]]:
    """
    Get train and test data loaders.
    :param dataset_path:
        The path to the via json file.
    :param train_transform:
        The transform to apply to the training data.
    :param test_transform:
        The transform to apply to the test data.
    :param batch_size:
        How many images to load at each call.
    :param test_percentage: 
        The percentage of the dataset to use for testing. The rest is used for training.
    :return:
        A tuple of train and test data loaders.
    """
    # use our dataset and defined transformations
    source_train = ViaDatasetOnlyAnnotated(
        dataset_path, transform=train_transform)
    source_test = ViaDatasetOnlyAnnotated(
        dataset_path, transform=test_transform)

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(source_train)).tolist()

    test_size = int(test_percentage * len(source_train))

    dataset_train = data.Subset(source_train, indices[:-test_size])
    dataset_test = data.Subset(source_test, indices[-test_size:])

    logging.info(f'Train dataset size: {len(dataset_train)}')
    logging.info(f'Test dataset size: {len(dataset_test)}')

    # define training and validation data loaders
    data_loader_train = DataLoader[DatasetItem](
        dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_into_lists)

    data_loader_test = DataLoader[DatasetItem](
        dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_into_lists)

    return data_loader_train, data_loader_test
