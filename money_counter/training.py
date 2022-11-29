from logging import getLogger
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from money_counter import engine, models
from money_counter.utils import get_device

logging = getLogger(__name__)


def train(
        model: nn.Module, model_name: str, data_loader_train: DataLoader, data_loader_test: DataLoader,
        *,
        num_epochs: int = 10, device: Optional[torch.device] = None,
        version_manager: Optional[models.VersionManager] = None, state_dir: Optional[str],
        evaluate: bool = False, print_frequency: int = 10) -> None:
    """
    Train a model on a dataset.
    :param model: The model to train.
    :param model_name: The name of the model.
    :param data_loader_train: A data loader containing the training data.
    :param data_loader_test: A data loader containing the test data.
    :param num_epochs: The number of epochs to train for.
    :param device: The device to use in training.
    :param version_manager: Used to save/load the model states to/from disk.
    :param state_dir: The directory where states are being saved to. This is only used if version_manager is None. 
    :param evaluate: Whether to evaluate the model after each epoch. 
    :param print_frequency: How many frequent to print the loss details.    
    """

    if device is None:
        device = get_device()

        logging.debug(f'Using device: {device}')

    if (version_manager is None and state_dir is None) or (version_manager is not None and state_dir is not None):
        raise Exception(
            "You must provide either a version manager or a state directory")

    if version_manager is None:
        version_manager = models.VersionManager(state_dir)

    # move model to the right device
    model = model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # TODO: Optimizer should be configurable    
    optimizer = torch.optim.SGD(params, lr=0.001, #lr=0.005,
                                momentum=0.9, nesterov=True) #weight_decay=0.0005)

    # TODO: Scheduler should be configurable
    # and a learning rate scheduler
    lr_scheduler = None
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                               step_size=3,
    #                                               gamma=0.1)

    # load the model if possible
    epoch, loss = version_manager.load_model(model_name, model, optimizer)

    epoch += 1

    logging.info(f"Starting training from epoch '{epoch}' with loss = '{loss}'.")

    while epoch < num_epochs:
        # train for one epoch, printing every 10 iterations
        metric_logger = engine.train_one_epoch(model, optimizer, data_loader_train,
                                               device, epoch, print_freq=print_frequency)
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        loss = metric_logger.meters['loss'].global_avg

        version_manager.save_model(model_name, model, optimizer, epoch, loss)

        if evaluate:
            engine.evaluate(model, data_loader_test, device=device)

        epoch += 1

    logging.info("That's it!")
