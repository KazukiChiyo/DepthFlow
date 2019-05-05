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
        self.loss_meter = AverageMeter()
        self.epe_meter = AverageMeter()

    def run_epoch(self):
        """Run an epoch of training."""
        self.model.train()
        # epoch_loss = 0
        # epoch_metric = 0

        for step, (input, target) in enumerate(tqdm(self.data_loader), 1):
            rgb_inputs = torch.cat((input[0], input[1]), 1).to(self.device)
            depth_inputs = torch.cat((input[2], input[3]), 1).to(self.device)
            #input = torch.cat(input, 1).to(self.device)
            target = target.to(self.device)
            outputs = self.model(rgb_inputs, depth_inputs)
            #outputs = self.model(input)
            b, _, h, w = target.size()
            outputs = [F.interpolate(outputs[0], (h, w)), *outputs[1:]]
            loss = self.criterion(outputs, target)
            self.loss_meter.update(loss.item(), b)
            epe = self.metric(outputs[0], target)
            self.epe_meter.update(epe.item(), b)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # epoch_loss += loss.item()*b
            # epoch_metric += metric.item()*b
            # count += b

        # return epoch_loss/count, epoch_metric/count
        return self.loss_meter.avg, self.epe_meter.avg


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
        self.epe_meter = AverageMeter()

    def run_epoch(self):
        """Run an epoch of validation."""
        self.model.eval()
        # epoch_metric = 0

        for step, (input, target) in enumerate(tqdm(self.data_loader), 1):
            #input = torch.cat(input, 1).to(self.device)
            rgb_inputs = torch.cat((input[0], input[1]), 1).to(self.device)
            depth_inputs = torch.cat((input[2], input[3]), 1).to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = self.model(rgb_inputs, depth_inputs)
                #output = self.model(input)
                epe = self.metric(output, target)
                self.epe_meter.update(epe.item(), target.size(0))

        #     epoch_metric += metric.item()*b
        #     count += b
        #
        # return epoch_metric/count
        return self.epe_meter.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)
