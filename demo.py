from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
import numpy as np
from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()


#model_path = '07-CSRNet_all_ep_39_mae_32.6_mse_74.3.pth'
#model_path = 'resnet101-5d3b4d8f.pth'
model_path = '11-ResSFCN-101_all_ep_94_mae_26.8_mse_66.1.pth'
#you need download the modelfile from
#https://mailnwpueducn-my.sharepoint.com/personal/gjy3035_mail_nwpu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fgjy3035%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2F%E8%AE%BA%E6%96%87%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%2FPreTrained%20Models%20on%20GCC%5Frd&originalPath=aHR0cHM6Ly9tYWlsbndwdWVkdWNuLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL2dqeTMwMzVfbWFpbF9ud3B1X2VkdV9jbi9FdkgxWWNkRkJiUkpvYzdHX1ZxdjBEd0JHbXV5WFBsSDg5OU9yUTU0LWN5YldRP3J0aW1lPU5pWHJISjZaMTBn


                     
def re_name_weight(weight_dict):
    #wts = torch.load('xxx.pth')
    new_wts = {}
    for i_key in weight_dict.keys():
        new_key= i_key.replace('module.','')
        print (new_key)
        new_wts[new_key] = weight_dict[i_key]
    return new_wts

def my_demo(file_list, model_path):
    Net_OK = ['Res101_SFCN','CSRNet']
    if(cfg.NET not in Net_OK):
        print('net is not Res101_SFCN CSRNet demo not work')
        return
    net = CrowdCounter(cfg.GPU_ID,cfg.NET)

    new_weight_dict = torch.load(model_path)
    if(cfg.GPU_ID == [0]):
        new_weight_dict = re_name_weight(new_weight_dict)
    net.load_state_dict(new_weight_dict)
    net.cuda()
    net.eval()
    print('net eval is ok=================')

    f1 = plt.figure(1)
    for filename in file_list:
        print( filename )
        img = Image.open(filename)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)
        with torch.no_grad():
            img = Variable(img[None,:,:,:]).cuda()
            pred_map = net.test_forward(img)
            density_pre = pred_map.squeeze().cpu().numpy() / 100.
            num_people = int(np.sum(density_pre))
            print('in this picture,there are ',num_people,' people')



if __name__ == '__main__':
    file_list = ['/home/deeplp/Downloads/A/people15.jpg']
    my_demo(file_list, model_path)




