import json
import os
from typing import (Callable, Dict, Generic, Iterable, Iterator, List,
                    Optional, Tuple, TypeVar, cast)

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

TTransformedImage = TypeVar("TTransformedImage")

ImageTransform = Callable[[Image.Image], TTransformedImage]
"""Callback to transform an image."""

TTransformedTarget = TypeVar("TTransformedTarget")

TargetTransform = Callable[[Target], TTransformedTarget]
"""Callback to transform a target."""

DatasetItem = Tuple[TTransformedImage, TTransformedTarget]
"""Item returned by a Torch Dataset instance."""


def _no_op_transform(x): return x


label_map = {label: i for i, label in enumerate(CLASSES)}


class ResizeImage:
    """Resize the image to either 3000x4000 or 4000x3000 depending on the aspect ratio."""

    def __call__(self, image: Image.Image) -> Image.Image:
        size = image.size
        if size[0] > size[1]:
            image = image.resize((4000, 3000))
        else:
            image = image.resize((3000, 4000))

        return image


class CoinsDataset(Dataset[DatasetItem], Generic[TTransformedImage, TTransformedTarget]):
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
            transform: ImageTransform = _no_op_transform, target_transform: TargetTransform = _no_op_transform,
            metadata_transform: Optional[Callable[[List[v2.ImageMetadata]], List[v2.ImageMetadata]]] = None):
        """
        Initialises a new instance of CoinsDataset.
        :param annotations_path: path to the json file produced by the VIA (VGG Image Annotation) tool.
        :param root_dir: directory with all the images.
        :param transform: optional transform to be applied on a sample.
        :param target_transform: optional transform to be applied on the target.
        """
        if not os.path.isfile(annotations_path):
            raise FileNotFoundError(f"File '{annotations_path}' not found.")
        
        self._root_dir = os.path.dirname(annotations_path) # Images are relative to the path of the annotations file
        """Directory with the images."""
        self._transform = transform
        """Additional transform to be applied on the image."""
        self._transform_metadata = target_transform
        """Additional transform to be applied on the metadata."""

        self.filename_map: Dict[str, int] = {}
        """The map between the filenames and the encoded filenames."""

        with open(annotations_path) as f:
            via: v2.ViaV2SaveFileFormat = json.load(f)

        # Get the metadata for the images
        self._images_metadata = [
            v for _, v in via['_via_img_metadata'].items()]

        if (metadata_transform is not None):
            self._images_metadata = metadata_transform(self._images_metadata)

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
        target = to_target(
            self._images_metadata[idx], image.size, label_map, self.filename_map)

        image = self._transform(image)

        target = self._transform_metadata(target)

        return image, target

    def _get_image(self, filename: str) -> Image.Image:
        """Get the image with the given name."""
        image_path = os.path.join(self._root_dir, filename)
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')

        return image


class CoinsDatasetOnlyAnnotated(CoinsDataset[TTransformedImage, TTransformedTarget]):
    """
    The dataset of coins, but only the annotated images.
    """

    def __init__(
            self, annotations_path: str, transform: ImageTransform = _no_op_transform, transform_target: TargetTransform = _no_op_transform):
        """
        Initialises a new instance of CoinsDataset.
        :param viajson_path: path to the json file produced by the VIA (VGG Image Annotation) tool
        :param root_dir: directory with all the images.
        :param transform: optional transform to be applied on a sample.
        """
        super().__init__(annotations_path, 
                         transform=transform, target_transform=transform_target)

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

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    ResizeImage()
])


def get_data_loaders(dataset_path: str, transform: ImageTransform = default_transform, batch_size=3):
    """Get train and test data loaders."""

    json_path = dataset_path

    # use our dataset and defined transformations
    source = CoinsDatasetOnlyAnnotated(
        dataset_path, transform=transform)
    #dataset_test = CoinsDatasetOnlyAnnotated(json_path, coins_imgs_dir, transform=transform)

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(source)).tolist()

    test_percentage = 0.2
    test_size = int(test_percentage * len(source))

    dataset_train = data.Subset(source, indices[:-test_size])
    dataset_test = data.Subset(source, indices[-test_size:])

    print(f'Train dataset size: {len(dataset_train)}')
    print(f'Test dataset size: {len(dataset_test)}')

    # define training and validation data loaders
    data_loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_into_lists)
    data_loader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_into_lists)

    return data_loader_train, data_loader_test
