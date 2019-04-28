import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from config import cfg


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name,loss_1_fn,loss_2_fn):
        super(CrowdCounter, self).__init__()
        if model_name == 'CMTL':
            from M2T2OCC_Model.CMTL import CMTL as net  

        self.CCN = net()
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = loss_1_fn.cuda()
        self.loss_bce_fn = loss_2_fn.cuda()

    @property
    def loss(self):
        return self.loss_mse, self.cross_entropy*cfg.LAMBDA_1


    def forward(self, img, gt_map=None, gt_cls_label=None):
        density_map, density_cls_score = self.CCN(img)

        # pdb.set_trace()

        density_cls_prob = F.softmax(density_cls_score,dim=1)

        self.loss_mse, self.cross_entropy = self.build_loss(density_map.squeeze(), gt_map.squeeze(), density_cls_prob, gt_cls_label)
        return density_map

    def build_loss(self, density_map, gt_data, density_cls_score, gt_cls_label):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        # pdb.set_trace()
        cross_entropy = self.loss_bce_fn(density_cls_score, gt_cls_label)
        return loss_mse, cross_entropy

    def test_forward(self, img):
        density_map, density_cls_score = self.CCN(img)
        return density_map

