# Author: Kexuan Zou
# Date: Mar 16, 2019
# License: MIT

import torch
import torch.nn.functional as F
from tqdm import tqdm

class Train(object):
    """Train the given model with data loader, optimizer, and loss criterion.
    Parameters:
    -----------
    model: torch.nn.Module
        Model object to train.
    data_loader: torch.utils.data.DataLoader
        Data loader that iterates over (input, label) pairs.
    optim: torch.optim
        Optimization algorithm.
    criterion: torch.nn
        The loss criterion.
    metric: torch.nn
        Metric evaluating the output.
    device: torch.device
        An object representing the device on which tensors are allocated.
    """
    def __init__(self, model, data_loader, optim, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self):
        """Run an epoch of training."""
        self.model.train()
        epoch_loss = 0
        epoch_metric = 0

        for step, (input, target) in enumerate(tqdm(self.data_loader), 1):
            inputs = torch.cat(input, 1).to(self.device)
            target = target.to(self.device)
            outputs = self.model(inputs)
            _, _, h, w = target.size()
            outputs = [F.interpolate(outputs[0], (h, w)), *outputs[1:]]
            loss = self.criterion(outputs, target)
            metric = self.metric(outputs[0], target)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            epoch_loss += loss.item()
            epoch_metric += metric.item()

        return epoch_loss/len(self.data_loader), epoch_metric/len(self.data_loader)


class Test(object):
    """Test the model with data loader and loss criterion.
    Parameters:
    -----------
    model: torch.nn.Module
        Model object to train.
    data_loader: torch.utils.data.DataLoader
        Data loader that iterates over (input, label) pairs.
    metric: torch.nn
        Metric evaluating the output.
    device: torch.device
        An object representing the device on which tensors are allocated.
    """
    def __init__(self, model, data_loader, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.metric = metric
        self.device = device

    def run_epoch(self):
        """Run an epoch of validation."""
        self.model.eval()
        epoch_metric = 0

        for step, (input, target) in enumerate(tqdm(self.data_loader), 1):
            inputs = torch.cat(input, 1).to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                metric = self.metric(outputs, target)

            epoch_metric += metric.item()

        return epoch_metric/len(self.data_loader)
