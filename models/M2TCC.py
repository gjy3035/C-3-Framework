import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from config import cfg

from misc import pytorch_ssim
from misc.ssim_loss import SSIM_Loss
class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()        
        
        if model_name == 'SANet':
            from M2TCC_Model.SANet import SANet as net


        self.CCN = net()
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        self.criterionSSIM = pytorch_ssim.SSIM(window_size=11).cuda()
        # self.criterionSSIM = SSIM_Loss(1).cuda()
        
    @property
    def loss(self):
        return self.loss_mse, self.loss_ssim*cfg.LAMBDA_1
    
    def forward(self, img, gt_map):                               
        density_map = self.CCN(img)                          
        self.loss_mse= self.loss_mse_fn(density_map.squeeze(), gt_map.squeeze())   
        # pdb.set_trace()
        self.loss_ssim= 1 - self.criterionSSIM(density_map, gt_map[:,None,:,:]) 
        # self.loss_ssim= self.criterionSSIM(density_map, gt_map[:,None,:,:])  

        return density_map


    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

