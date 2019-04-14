# Author: Kexuan Zou
# Date: Mar 22, 2019
# License: MIT
# Reference: https://github.com/NVIDIA/flownet2-pytorch/blob/master/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def _EPE(input_flow, target_flow, mean=True):
    EPE_map = torch.norm(target_flow - input_flow, p=2, dim=1)
    mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
    EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/EPE_map.size(0)


def sparse_max_pool2d(input, size):
    positive = (input > 0).float()
    negative = (input < 0).float()
    return F.adaptive_max_pool2d(input*positive, size) - F.adaptive_max_pool2d(-input*negative, size)


class MultiScaleEPE(nn.Module):
    def __init__(self, n_scales=5, l_weight=0.32):
        super(MultiScaleEPE, self).__init__()

        self.n_scales = n_scales
        self.loss_weights = [(l_weight/2**scale) for scale in range(self.n_scales)]

    def one_scale(self, output, target):
        _, _, h, w = output.size()
        target_scaled = sparse_max_pool2d(target, (h, w))
        return _EPE(output, target_scaled, mean=False)

    def forward(self, output, target):
        loss = 0
        target = target
        for i, output_ in enumerate(output):
            loss += self.loss_weights[i]*self.one_scale(output_, target)
        return loss


class EPE(nn.Module):
    def __init__(self, div_flow=0.05):
        self.div_flow = div_flow

    def __call__(self, output, target):
        _, _, h, w = target.size()
        output_scaled = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
        return _EPE(output_scaled, target)
