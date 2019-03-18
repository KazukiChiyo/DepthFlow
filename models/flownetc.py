import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util import conv, predict_flow, deconv, crop_like, correlate


class FlowNetC(nn.Module):
    expansion = 1

    def __init__(self, BN=True):
        super(FlowNetC, self).__init__()

        self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.conv1 = Conv2DNorm(3, 64, kernel_size=7, stride=2, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)
        self.conv2 = Conv2DNorm(64, 128, kernel_size=5, stride=2, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)
        self.conv3 = Conv2DNorm(128, 256, kernel_size=5, stride=2, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)
        self.conv_redir = Conv2DNorm(256, 32, kernel_size=1, stride=1, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)

        self.conv3_1 = Conv2DNorm(473, 256, kernel_size=3, stride=1, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)
        self.conv4   = Conv2DNorm(256, 512, kernel_size=3, stride=2, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)
        self.conv4_1 = Conv2DNorm(512, 512, kernel_size=3, stride=1, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)
        self.conv5   = Conv2DNorm(512, 512, kernel_size=3, stride=2, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)
        self.conv5_1 = Conv2DNorm(512, 512, kernel_size=3, stride=1, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)
        self.conv6   = Conv2DNorm(512, 1024, kernel_size=3, stride=2, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)
        self.conv6_1 = Conv2DNorm(1024, 1024, kernel_size=3, stride=1, kernel_initializer='kaiming', batch_norm=batch_norm, activation=self.activ)

        self.deconv5 = Deconv2DNorm(1024, 512, kernel_size=4, stride=2, bias=False, kernel_initializer='kaiming', activation=self.activ)
        self.deconv4 = Deconv2DNorm(1026, 256, kernel_size=4, stride=2, bias=False, kernel_initializer='kaiming', activation=self.activ)
        self.deconv3 = Deconv2DNorm(770, 128, kernel_size=4, stride=2, bias=False, kernel_initializer='kaiming', activation=self.activ)
        self.deconv2 = Deconv2DNorm(386, 64, kernel_size=4, stride=2, bias=False, kernel_initializer='kaiming', activation=self.activ)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = Deconv2DNorm(2, 2, 4, stride=2, bias=False)
        self.upsampled_flow5_to_4 = Deconv2DNorm(2, 2, 4, stride=2, bias=False)
        self.upsampled_flow4_to_3 = Deconv2DNorm(2, 2, 4, stride=2, bias=False)
        self.upsampled_flow3_to_2 = Deconv2DNorm(2, 2, 4, stride=2, bias=False)

        for m in self.modules():
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        x1 = x[:,:3]
        x2 = x[:,3:]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        out_conv_redir = self.conv_redir(out_conv3a)
        out_correlation = correlate(out_conv3a,out_conv3b)

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(in_conv3_1)
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
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2a)

        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownetc(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetC(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def flownetc_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetC(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
