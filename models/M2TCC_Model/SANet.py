#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import initialize_weights
import pdb


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn=False):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=not self.use_bn)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        # pdb.set_trace()
        x = self.tconv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class SAModule_Head(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule_Head, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=1)
        self.branch3x3 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=3, padding=1)
        self.branch5x5 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=5, padding=2)
        self.branch7x7 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=7, padding=3)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SAModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=1)
        self.branch3x3 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=3, padding=1),
                        )
        self.branch5x5 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=5, padding=2),
                        )
        self.branch7x7 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=7, padding=3),
                        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SANet(nn.Module):
    def __init__(self, gray_input=False, use_bn=True):
        super(SANet, self).__init__()
        if gray_input:
            in_channels = 1
        else:
            in_channels = 3

        self.encoder = nn.Sequential(
            SAModule_Head(in_channels, 64, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(64, 128, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(128, 128, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(128, 128, use_bn),
            )

        self.decoder = nn.Sequential(
            BasicConv(128, 64, use_bn=use_bn, kernel_size=9, padding=4),
            BasicDeconv(64, 64, 2, stride=2, use_bn=use_bn),
            BasicConv(64, 32, use_bn=use_bn, kernel_size=7, padding=3),
            BasicDeconv(32, 32, 2, stride=2, use_bn=use_bn),
            BasicConv(32, 16,  use_bn=use_bn, kernel_size=5, padding=2),
            BasicDeconv(16, 16, 2, stride=2, use_bn=use_bn),
            BasicConv(16, 16,  use_bn=use_bn, kernel_size=3, padding=1),
            BasicConv(16, 1, use_bn=False, kernel_size=1),
            )
        initialize_weights(self.modules())

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out