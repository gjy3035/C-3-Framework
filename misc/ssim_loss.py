#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _assert_no_grad, _Loss
import numpy as np

def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    kernel = np.exp(-0.5*(x*x+y*y)/(sigma*sigma))
    kernel /= kernel.sum()
    return kernel

class SSIM_Loss(_Loss):
    def __init__(self, in_channels, size=11, sigma=1.5, size_average=True):
        super(SSIM_Loss, self).__init__(size_average)
        #assert in_channels == 1, 'Only support single-channel input'
        self.in_channels = in_channels
        self.size = int(size)
        self.sigma = sigma
        self.size_average = size_average

        kernel = gaussian_kernel(self.size, self.sigma)
        self.kernel_size = kernel.shape
        weight = np.tile(kernel, (in_channels, 1, 1, 1))
        self.weight = Parameter(torch.from_numpy(weight).float(), requires_grad=False)

    def forward(self, input, target, mask=None):
        _assert_no_grad(target)
        mean1 = F.conv2d(input, self.weight, padding=self.size, groups=self.in_channels)
        mean2 = F.conv2d(target, self.weight, padding=self.size, groups=self.in_channels)
        mean1_sq = mean1*mean1
        mean2_sq = mean2*mean2
        mean_12 = mean1*mean2

        sigma1_sq = F.conv2d(input*input, self.weight, padding=self.size, groups=self.in_channels) - mean1_sq
        sigma2_sq = F.conv2d(target*target, self.weight, padding=self.size, groups=self.in_channels) - mean2_sq
        sigma_12 = F.conv2d(input*target, self.weight, padding=self.size, groups=self.in_channels) - mean_12
    
        C1 = 0.01**2
        C2 = 0.03**2

        ssim = ((2*mean_12+C1)*(2*sigma_12+C2)) / ((mean1_sq+mean2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        if self.size_average:
            out = 1 - ssim.mean()
        else:
            out = 1 - ssim.view(ssim.size(0), -1).mean(1)
        return out


if __name__ == '__main__':
    data = torch.zeros(1, 1, 1, 1)
    data += 0.001
    target = torch.zeros(1, 1, 1, 1)
    data = Variable(data, requires_grad=True)
    target = Variable(target)

    model = SSIM_Loss(1)
    loss = model(data, target) 
    loss.backward()
    print(loss)
    print(data.grad)