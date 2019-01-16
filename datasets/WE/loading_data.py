import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from WE import WE
from setting import cfg_data 
import torch
import random


def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    train_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
        own_transforms.RandomHorizontallyFlip()
    ])
    val_main_transform = None
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = WE(cfg_data.DATA_PATH+'/train', 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=True)


    test_name = cfg_data.VAL_FOLDER 
    
    val_loader = []

    for subname in test_name:
        sub_set = WE(cfg_data.DATA_PATH+'/test/' + subname, 'test', main_transform=val_main_transform, img_transform=img_transform, gt_transform=gt_transform)
        val_loader.append(DataLoader(sub_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=True)) 


    return train_loader, val_loader, restore_transform
