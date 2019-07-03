import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .SHHA import SHHA
from .setting import cfg_data 
import torch
import random

def get_min_size(batch):

    min_ht = cfg_data.TRAIN_SIZE[0]
    min_wd = cfg_data.TRAIN_SIZE[1]

    for i_sample in batch:
        
        _,ht,wd = i_sample.shape
        if ht<min_ht:
            min_ht = ht
        if wd<min_wd:
            min_wd = wd
    return min_ht,min_wd

def random_crop(img,den,dst_size):
    # dst_size: ht, wd

    _,ts_hd,ts_wd = img.shape

    x1 = random.randint(0, ts_wd - dst_size[1])//cfg_data.LABEL_FACTOR*cfg_data.LABEL_FACTOR
    y1 = random.randint(0, ts_hd - dst_size[0])//cfg_data.LABEL_FACTOR*cfg_data.LABEL_FACTOR
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]

    label_x1 = x1//cfg_data.LABEL_FACTOR
    label_y1 = y1//cfg_data.LABEL_FACTOR
    label_x2 = x2//cfg_data.LABEL_FACTOR
    label_y2 = y2//cfg_data.LABEL_FACTOR

    return img[:,y1:y2,x1:x2], den[label_y1:label_y2,label_x1:label_x2]

def share_memory(batch):
    out = None
    if False:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out

def SHHA_collate(batch):
    # @GJY 
    r"""Puts each data field into a tensor with outer dimension batch size"""

    transposed = list(zip(*batch)) # imgs and dens
    imgs, dens = [transposed[0],transposed[1]]


    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):
        
        min_ht, min_wd = get_min_size(imgs)

        # print min_ht, min_wd

        # pdb.set_trace()
        
        cropped_imgs = []
        cropped_dens = []
        for i_sample in range(len(batch)):
            _img, _den = random_crop(imgs[i_sample],dens[i_sample],[min_ht,min_wd])
            cropped_imgs.append(_img)
            cropped_dens.append(_den)


        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))

        return [cropped_imgs,cropped_dens]

    raise TypeError((error_msg.format(type(batch[0]))))


def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    factor = cfg_data.LABEL_FACTOR
    train_main_transform = own_transforms.Compose([
    	own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(factor),
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = SHHA(cfg_data.DATA_PATH+'/train', 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader =None
    if cfg_data.TRAIN_BATCH_SIZE==1:
        train_loader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=True, drop_last=True)
    elif cfg_data.TRAIN_BATCH_SIZE>1:
        train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8, collate_fn=SHHA_collate, shuffle=True, drop_last=True)
    
    

    val_set = SHHA(cfg_data.DATA_PATH+'/test', 'test', main_transform=None, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    return train_loader, val_loader, restore_transform
