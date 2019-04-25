from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *

# model_path = '../PyTorch_Pretrained/alexnet-owt-4df8aa71.pth'

class AlexNet(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNet, self).__init__()
        alex = models.alexnet(pretrained=pretrained)
        # if pretrained:
        #     alex.load_state_dict(torch.load(model_path))
        features = list(alex.features.children())
        
        self.layer1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=4) # original padding is 4
        self.layer1plus = nn.Sequential(nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Conv2d(64, 192, kernel_size=5, padding=3) # original padding is 2
        self.layer2plus_to_5 = nn.Sequential(*features[4:12])
        self.de_pred = nn.Sequential(Conv2d(256, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))


        self.layer1.load_state_dict(alex.features[0].state_dict())
        self.layer2.load_state_dict(alex.features[3].state_dict())



    def forward(self, x):
        x = self.layer1(x) 
        x = self.layer1plus(x)  
        x = self.layer2(x)
        x = self.layer2plus_to_5(x)  
        x = self.de_pred(x)

        x = F.upsample(x,scale_factor=16)

        return x