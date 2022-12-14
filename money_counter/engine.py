import math
import sys
from logging import getLogger
from typing import List, cast, Optional as Opt

import torch
import torchvision
import torchvision.models.detection.mask_rcnn
from torch.utils.data import DataLoader

from .coco_eval import CocoEvaluator, IoUType
from .coco_utils import get_coco_api_from_dataset
from .models import PredictedTarget
from .utils import MetricLogger, SmoothedValue, Timer, get_device, reduce_dict, PrintToLog

logging = getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        device: Opt[torch.device],
        epoch: int,
        print_freq: int = 10,
        scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    if device == None:
        device = get_device()

    lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        losses_reduced = cast(torch.Tensor, losses_reduced)

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            logging.error(f"Loss is {loss_value}, stopping training")
            logging.error(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model) -> List[IoUType]:
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types: List[IoUType] = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: Opt[torch.device] = None, *, print_freq: int = 100, coco_evaluator: Opt[CocoEvaluator] = None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    if coco_evaluator is None:
        logging.info("Loading COCO API...")

        with PrintToLog(__name__):
            coco = get_coco_api_from_dataset(data_loader.dataset)
            iou_types = _get_iou_types(model)
            coco_evaluator = CocoEvaluator(coco, iou_types)

        logging.info("Loaded COCO API.")

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with Timer() as model_time:
            outputs = model(images)

        # Move output to the CPU
        outputs = [{k: v.to(cpu_device) for k, v in output.items()}
                   for output in outputs]

        outputs = {target["image_id"].item(): output
                   for target, output in zip(targets, outputs)}

        with Timer() as evaluator_time:
            coco_evaluator.update(outputs)

        metric_logger.update(model_time=model_time.elapsed(),
                             evaluator_time=evaluator_time.elapsed())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    logging.info("Averaged stats:", metric_logger)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    torch.set_num_threads(n_threads)

    return coco_evaluator


def apply_nms(predicted_targets: List[PredictedTarget], nms_threshold: float = 0.5):
    """
    Applies non-maximum suppression to the predicted targets.
    """
    for i, predicted_target in enumerate(predicted_targets):
        boxes = predicted_target['boxes']
        scores = predicted_target['scores']
        labels = predicted_target['labels']
        keep = torchvision.ops.nms(boxes, scores, nms_threshold)
        predicted_targets[i]['boxes'] = boxes[keep]
        predicted_targets[i]['scores'] = scores[keep]
        predicted_targets[i]['labels'] = labels[keep]
