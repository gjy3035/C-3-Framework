import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from models.M2TCC import CrowdCounter
from config import cfg
from misc.utils import *
import pdb


class Trainer():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET

        if self.net_name in ['SANet']:
            loss_1_fn = nn.MSELoss()
            from misc import pytorch_ssim
            loss_2_fn = pytorch_ssim.SSIM(window_size=11)

        self.net = CrowdCounter(cfg.GPU_ID,self.net_name,loss_1_fn,loss_2_fn).cuda()
        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)          

        self.train_record = {'best_mae': 1e20, 'best_mse':1e20, 'best_model_name': ''}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 


        self.epoch = 0
        self.i_tb = 0
        
        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        self.train_loader, self.val_loader, self.restore_transform = dataloader()

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)


    def forward(self):

        # self.validate_V3()
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()
                
            # training    
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

            # validation
            if epoch%cfg.VAL_FREQ==0 or epoch>cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                elif self.data_mode is 'WE':
                    self.validate_V2()
                elif self.data_mode is 'GCC':
                    self.validate_V3()
                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )


    def train(self): # training for all datasets
        self.net.train()
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            img, gt_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            self.optimizer.zero_grad()
            pred_map = self.net(img, gt_map)
            loss1,loss2 = self.net.loss
            loss = loss1+loss2
            loss.backward()
            self.optimizer.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.writer.add_scalar('train_loss1', loss1.item(), self.i_tb)
                self.writer.add_scalar('train_loss2', loss2.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print( '[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' % \
                        (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff) )
                print( '        [cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data/self.cfg_data.LOG_PARA, pred_map[0].sum().data/self.cfg_data.LOG_PARA) )





    def validate_V1(self):# validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data


            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.net.forward(img,gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()
                
                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    loss1,loss2 = self.net.loss
                    loss = loss1.item()+loss2.item()
                    losses.update(loss)
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)
        print_summary(self.exp_name,[mae, mse, loss],self.train_record)


    def validate_V2(self):# validate_V2 for WE

        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)

        
        for i_sub,i_loader in enumerate(self.val_loader,0):

            for vi, data in enumerate(i_loader, 0):
                img, gt_map = data

                with torch.no_grad():
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()

                    pred_map = self.net.forward(img,gt_map)

                    pred_map = pred_map.data.cpu().numpy()
                    gt_map = gt_map.data.cpu().numpy()
                    
                    
                    for i_img in range(pred_map.shape[0]):
                        pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                        gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                        losses.update(self.net.loss.item(),i_sub)
                        maes.update(abs(gt_count-pred_cnt),i_sub)
                    if vi==0:
                        vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = np.average(maes.avg)
        loss = np.average(losses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, 0, loss],self.train_record,self.log_txt)
        print_summary(self.exp_name,[mae, 0, loss],self.train_record)


    def validate_V3(self):# validate_V3 for GCC

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}
        c_mses = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}


        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map, attributes_pt = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()


                pred_map = self.net.forward(img,gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()
                
                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map)/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map)/self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count-pred_cnt)
                    s_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)

                    loss1,loss2 = self.net.loss
                    loss = loss1.item()+loss2.item()
                    losses.update(loss)
                    maes.update(s_mae)
                    mses.update(s_mse)   
                    attributes_pt = attributes_pt.squeeze() 
                    c_maes['level'].update(s_mae,attributes_pt[0])
                    c_mses['level'].update(s_mse,attributes_pt[0])
                    c_maes['time'].update(s_mae,attributes_pt[1]/3)
                    c_mses['time'].update(s_mse,attributes_pt[1]/3)
                    c_maes['weather'].update(s_mae,attributes_pt[2])
                    c_mses['weather'].update(s_mse,attributes_pt[2])


                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)


        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)

        print_GCC_summary(self.log_txt,self.epoch,[mae, mse, loss],self.train_record,c_maes,c_mses)