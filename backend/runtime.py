import torch
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
    device: torch.device
        An object representing the device on which tensors are allocated.
    metric: function
        An instance specifying the metric to return.
    """
    def __init__(self, model, data_loader, optim, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.device = device

    def run_epoch(self):
        """Run an epoch of training."""
        self.model.train()
        epoch_loss = 0.0
        for step, batch_data in enumerate(tqdm(self.data_loader), 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            epoch_loss += loss.item()

        return epoch_loss/len(self.data_loader)


class Test(object):
    """Test the model with data loader and loss criterion.
    Parameters:
    -----------
    model: torch.nn.Module
        Model object to train.
    data_loader: torch.utils.data.DataLoader
        Data loader that iterates over (input, label) pairs.
    criterion: torch.nn
        The loss criterion.
    device: torch.device
        An object representing the device on which tensors are allocated.
    metric: function
        An instance specifying the metric to return.
    """
    def __init__(self, model, data_loader, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device

    def run_epoch(self):
        """Run an epoch of validation."""
        self.model.eval()
        epoch_loss = 0.0
        for step, batch_data in enumerate(tqdm(self.data_loader), 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            epoch_loss += loss.item()

        return epoch_loss/len(self.data_loader)
