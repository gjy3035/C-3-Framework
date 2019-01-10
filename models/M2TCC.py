import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from config import cfg


class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name,loss_1_fn,loss_2_fn):
        super(CrowdCounter, self).__init__()        
        
        if model_name == 'SANet':
            from M2TCC_Model.SANet import SANet as net


        self.CCN = net()
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
        self.loss_1_fn = loss_1_fn.cuda()
        self.loss_2_fn = loss_2_fn.cuda()
        
    @property
    def loss(self):
        return self.loss_1, self.loss_2*cfg.LAMBDA_1
    
    def forward(self, img, gt_map):                               
        density_map = self.CCN(img)                          
        self.loss_1= self.loss_1_fn(density_map.squeeze(), gt_map.squeeze())   
        self.loss_2= 1 - self.loss_2_fn(density_map, gt_map[:,None,:,:]) 

        return density_map


    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

