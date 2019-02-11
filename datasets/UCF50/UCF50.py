import numpy as np
import os
import random
import pandas as pd
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps


class UCF50(data.Dataset):
    def __init__(self, data_path, folder, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.mode = mode

        self.img_files = []
        self.gt_files = []
        for i_folder in folder:
            folder_img = self.img_path + '/' + str(i_folder)
            folder_gt = self.gt_path + '/' + str(i_folder)
            for filename in os.listdir(folder_img):
                if os.path.isfile(os.path.join(folder_img,filename)):
                    self.img_files.append(folder_img + '/' + filename)
                    self.gt_files.append(folder_gt + '/' + filename.split('.')[0] + '.csv')   

        self.num_samples = len(self.img_files) 

        self.mode = mode
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        
        
    
    def __getitem__(self, index):

        img, den = self.read_image_and_gt(index)
      
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.gt_transform is not None:
            den = self.gt_transform(den)      
            
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,index):
        img = Image.open(os.path.join(self.img_path,self.img_files[index]))
        if img.mode == 'L':
            img = img.convert('RGB')

        den = pd.read_csv(os.path.join(self.gt_path,self.gt_files[index]), sep=',',header=None).values
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        
        return img, den


    def get_num_samples(self):
        return self.num_samples       
            
        
