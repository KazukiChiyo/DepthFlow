# Author: Kexuan Zou
# Date: Mar 13, 2019
# License: MIT
# Reference: https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/FlowNetS.py

import torch
import torch.nn as nn
from .layers import Conv2DNorm, Deconv2DNorm, crop_like


class DepthFlowNetS(nn.Module):

    def __init__(self, in_channels=6, depth_channels=2, batch_norm=True):
        super(DepthFlowNetS,self).__init__()

        self.conv1   = Conv2DNorm(in_channels, 64, 7, stride=2, padding=3, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.depth_conv1 = Conv2DNorm(depth_channels, 64, 7, stride=2, padding=3, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.conv1x1 = Conv2DNorm(128, 64, 1, stride=1, padding=0, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.conv2   = Conv2DNorm(64, 128, 5, stride=2, padding=2, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.conv3   = Conv2DNorm(128, 256, 5, stride=2, padding=2, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.conv3_1 = Conv2DNorm(256, 256, 3, padding=1, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.conv4   = Conv2DNorm(256, 512, 3, stride=2, padding=1, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.conv4_1 = Conv2DNorm(512, 512, 3, padding=1, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.conv5   = Conv2DNorm(512, 512, 3, stride=2, padding=1, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.conv5_1 = Conv2DNorm(512, 512, 3, padding=1, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.conv6   = Conv2DNorm(512, 1024, 3, stride=2, padding=1, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)
        self.conv6_1 = Conv2DNorm(1024, 1024, 3, padding=1, bias=False, kernel_initializer='kaiming', batch_norm=batch_norm, activation=True)

        self.deconv5 = Deconv2DNorm(1024, 512, 4, stride=2, padding=1, bias=False, kernel_initializer='kaiming', activation=True)
        self.deconv4 = Deconv2DNorm(1026, 256, 4, stride=2, padding=1, bias=False, kernel_initializer='kaiming', activation=True)
        self.deconv3 = Deconv2DNorm(770, 128, 4, stride=2, padding=1, bias=False, kernel_initializer='kaiming', activation=True)
        self.deconv2 = Deconv2DNorm(386, 64, 4, stride=2, padding=1, bias=False, kernel_initializer='kaiming', activation=True)

        self.predict_flow6 = Conv2DNorm(1024, 2, 3, padding=1, bias=False)
        self.predict_flow5 = Conv2DNorm(1026, 2, 3, padding=1, bias=False)
        self.predict_flow4 = Conv2DNorm(770, 2, 3, padding=1, bias=False)
        self.predict_flow3 = Conv2DNorm(386, 2, 3, padding=1, bias=False)
        self.predict_flow2 = Conv2DNorm(194, 2, 3, padding=1, bias=False)

        self.upsampled_flow6_to_5 = Deconv2DNorm(2, 2, 4, stride=2, padding=1, bias=False)
        self.upsampled_flow5_to_4 = Deconv2DNorm(2, 2, 4, stride=2, padding=1, bias=False)
        self.upsampled_flow4_to_3 = Deconv2DNorm(2, 2, 4, stride=2, padding=1, bias=False)
        self.upsampled_flow3_to_2 = Deconv2DNorm(2, 2, 4, stride=2, padding=1, bias=False)


    def forward(self, x, depth_x):
        out_conv1 = self.conv1(x)
        out_depth_conv1 = self.depth_conv1(depth_x)
        out_stacked_conv1 = torch.cat((out_conv1, out_depth_conv1), dim=1)
        out_combined_conv1 = self.conv1x1(out_stacked_conv1)

        out_conv2 = self.conv2(out_combined_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]


    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
