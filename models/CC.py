import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()        
        
        if model_name == 'vgg':
            from vgg import VGG as net
        elif model_name == 'vgg_encoder':
            from VGG_decoder import VGG as net
        elif model_name == 'csr':
            from csr_ori import CSRNet as net
        elif model_name == 'test':
            from test_model import test_model as net
        elif model_name == 'drn_d_38':
            from drn import drn_d_38 as net
        elif model_name == 'MCNN':
            from MCNN import MCNN as net
        elif model_name == 'MCNN_v1':
            from MCNN_v1 import MCNN as net
        elif model_name == 'MCNN_v2':
            from MCNN_v2 import MCNN as net
        elif model_name == 'MCNN_v3':
            from MCNN_v3 import MCNN as net
        elif model_name == 'MCNN_large':
            from MCNN_large import MCNN as net
        elif model_name == 'MCNN_big':
            from MCNN_big import MCNN as net
        elif model_name == 'MCNN_v4':
            from MCNN_v4 import MCNN as net
        elif model_name == 'MCNN_v5':
            from MCNN_v5 import MCNN as net
        elif model_name == 'res50':
            from res50 import res50 as net
        self.CCN = net()
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        
    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map):                               
        density_map = self.CCN(img)                          
        self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())               
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

