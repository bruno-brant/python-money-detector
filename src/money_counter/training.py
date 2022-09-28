from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from money_counter import engine, models


def train(
        model: nn.Module, model_name: str, data_loader_train: DataLoader, data_loader_test: DataLoader, *,
        num_epochs: int = 10, device: Optional[torch.device] = None,
        version_manager: Optional[models.VersionManager] = None, state_dir: Optional[str],
        evaluate: bool = False, print_frequency: int = 10):

    if device is None:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        print(f'Using device: {device}')

    if (version_manager is None and state_dir is None) or (version_manager is not None and state_dir is not None):
        raise Exception(
            "You must provide either a version manager or a state directory")

    if version_manager is None:
        version_manager = models.VersionManager(state_dir)

    # move model to the right device
    model = model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # load the model if possible
    epoch, loss = version_manager.load_model(model_name, model, optimizer)

    print(f"Starting training from epoch '{epoch}' with loss = '{loss}'.")

    while epoch < num_epochs:
        # train for one epoch, printing every 10 iterations
        metric_logger = engine.train_one_epoch(model, optimizer, data_loader_train,
                                        device, epoch, print_freq=print_frequency)
        # update the learning rate
        lr_scheduler.step()
        
        loss = metric_logger.meters['loss'].global_avg

        version_manager.save_model(model_name, model, optimizer, epoch, loss)

        if evaluate:
            engine.evaluate(model, data_loader_test, device=device)

        epoch += 1

    print("That's it!")
