import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .GCC import GCC
from .setting import cfg_data 
import torch
import random



def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    train_main_transform = own_transforms.Compose([
        # own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
    	own_transforms.RandomHorizontallyFlip()
    ])
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

    if cfg_data.VAL_MODE=='rd':
        test_list = 'test_list.txt'
        train_list = 'train_list.txt'
    elif cfg_data.VAL_MODE=='cc':
        test_list = 'cross_camera_test_list.txt'
        train_list = 'cross_camera_train_list.txt'
    elif cfg_data.VAL_MODE=='cl':
        test_list = 'cross_location_test_list.txt'
        train_list = 'cross_location_train_list.txt'    


    train_set = GCC(cfg_data.DATA_PATH+'/txt_list/' + train_list, 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=True)

    val_set = GCC(cfg_data.DATA_PATH+'/txt_list/'+ test_list, 'test', main_transform=None, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    return train_loader, val_loader, restore_transform
