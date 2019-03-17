# Author: Kexuan Zou
# Date: Nov 13, 2018; revision: Mar 13, 2019
# License: MIT

import torch.nn as nn
import torch.nn.init as init

activation_functions = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'softplus': nn.Softplus,
    'softmax': nn.Softmax
}

init_gain = {
    'relu': 1.41414,
    'leaky_relu': 1.41414,
    'elu': 1.41414,
    'sigmoid': 1,
    'tanh': 1.66667,
    'softplus': 1,
    'softmax': 1
}


class _LayerNd(nn.Module):
    def __init__(self, kernel_initializer, activation):
        super(_LayerNd, self).__init__()

        if isinstance(activation, str):
            self.activation = activation_functions[activation]()
        else:
            self.activation = activation

        if isinstance(kernel_initializer, str):
            if kernel_initializer == 'normal':
                self.kernel_initializer = init.normal_
            elif kernel_initializer == 'kaiming':
                self.kernel_initializer = init.kaiming_normal_
            elif kernel_initializer == 'xavier':
                self.kernel_initializer = init.xavier_normal_
                self.gain = init_gain.setdefault(activation, 1)
            elif kernel_initializer == 'orthogonal':
                self.kernel_initializer = init.orthogonal_
                self.gain = init_gain.setdefault(activation, 1)
        else:
            self.kernel_initializer = kernel_initializer


class Conv2DNorm(_LayerNd):
    """Applies 2D convolution over an input signal with batch normalization and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, kernel_initializer='normal', batch_norm=False, activation=None):
        super(Conv2DNorm, self).__init__(kernel_initializer=kernel_initializer, activation=activation)

        conv_base = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        if hasattr(self, 'gain'):
            self.kernel_initializer(conv_base.weight, gain=self.gain)
        else:
            self.kernel_initializer(conv_base.weight)

        if batch_norm:
            if activation:
                self.conv = nn.Sequential(
                    conv_base,
                    nn.BatchNorm2d(num_features=out_channels),
                    self.activation)
            else:
                self.conv = nn.Sequential(
                    conv_base,
                    nn.BatchNorm2d(num_features=out_channels))
        else:
            if activation:
                self.conv = nn.Sequential(
                    conv_base,
                    self.activation)
            else:
                self.conv = nn.Sequential(
                    conv_base)

    def forward(self, x):
        x = self.conv(x)
        return x


# reference: https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
class SeparableConv2D(_LayerNd):
    """Applies depthwise separable 2D convolution over an input signal with batch normalization and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, kernel_initializer='normal', batch_norm=False, activation=None):
        super(SeparableConv2D, self).__init__(kernel_initializer=kernel_initializer, activation=activation)

        conv_depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias)

        conv_pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias)

        init.xavier_normal_(conv_depthwise.weight)
        if hasattr(self, 'gain'):
            self.kernel_initializer(conv_pointwise.weight, gain=self.gain)
        else:
            self.kernel_initializer(conv_pointwise.weight)

        if batch_norm:
            if activation:
                self.conv = nn.Sequential(
                    conv_depthwise,
                    conv_pointwise,
                    nn.BatchNorm2d(num_features=out_channels),
                    self.activation)
            else:
                self.conv = nn.Sequential(
                    conv_depthwise,
                    conv_pointwise,
                    nn.BatchNorm2d(num_features=out_channels))
        else:
            if activation:
                self.conv = nn.Sequential(
                    conv_depthwise,
                    conv_pointwise,
                    self.activation)
            else:
                self.conv = nn.Sequential(
                    conv_depthwise,
                    conv_pointwise)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvResidual2D(Conv2DNorm):
    """Convolutional 2D residual block with batch normalization and activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, kernel_initializer='normal', batch_norm=False, activation=None):
        super(ConvResidual2D, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, kernel_initializer=kernel_initializer, batch_norm=batch_norm, activation=activation)

    def forward(self, x):
        out = self.conv(x)
        return x + out


class Deconv2DNorm(_LayerNd):
    """Applies 2D transposed convolution over an input signal with batch normalization and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, kernel_initializer='normal', dilation=1, batch_norm=False, activation=None):
        super(Deconv2DNorm, self).__init__(kernel_initializer=kernel_initializer, activation=activation)

        deconv_base = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation)

        if hasattr(self, 'gain'):
            self.kernel_initializer(deconv_base.weight, gain=self.gain)
        else:
            self.kernel_initializer(deconv_base.weight)

        if batch_norm:
            if activation:
                self.deconv = nn.Sequential(
                    deconv_base,
                    nn.BatchNorm2d(num_features=out_channels),
                    self.activation)
            else:
                self.deconv = nn.Sequential(
                    deconv_base,
                    nn.BatchNorm2d(num_features=out_channels))
        else:
            if activation:
                self.deconv = nn.Sequential(
                    deconv_base,
                    self.activation)
            else:
                self.deconv = nn.Sequential(
                    deconv_base)

    def forward(self, x):
        x = self.deconv(x)
        return x


def crop_like(input, target):
    """Crop input hieght and width to match target."""
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]
