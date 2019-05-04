import torch.utils.data as data
from .utils import default_loader

class ListDataset(data.Dataset):
    def __init__(self, dPath, path_list, rgb_transform=None, target_transform=None, depth_transform=None,
                 co_transform=None, loader=default_loader, depth=True):

        self.dPath = dPath
        self.path_list = path_list
        self.rgb_transform = rgb_transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader
        self.depth=depth
        self.depth_transform=depth_transform

    def __getitem__(self, index):
        inputs, target = self.path_list[index] # inputs: ['00001_img1.ppm', '00001_img2.ppm'], target:  '00001_flow.flo'
        # inputs=[img1, img2], targets=(384,512,2) flow data in ndarray
        inputs, target = self.loader(root=self.dPath, path_imgs=inputs, path_flo=target, depth=self.depth)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.rgb_transform is not None:
            inputs[0]=self.rgb_transform(inputs[0])
            inputs[1] = self.rgb_transform(inputs[1])
        if self.depth and self.depth_transform is not None:

            inputs[2]=self.depth_transform(inputs[2])
            inputs[3] = self.depth_transform(inputs[3])

        # order matters????????????
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)
