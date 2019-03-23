import torch.utils.data as data
from datasets.util import *

class ListDataset(data.Dataset):
    def __init__(self, dPath, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=default_loader):

        self.dPath = dPath
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        inputs, target = self.path_list[index] # inputs: ['00001_img1.ppm', '00001_img2.ppm'], target:  '00001_flow.flo'
        # inputs=[img1, img2], targets=(384,512,2) flow data in ndarray
        inputs, target = self.loader(self.dPath, inputs, target)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)
