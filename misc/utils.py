import numpy as np
import os
import math
import time
import random
import shutil

import torch
from torch import nn


import torchvision.utils as vutils
import torchvision.transforms as standard_transforms

import pdb


def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):    
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print( m )

def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():            
                if isinstance(m, nn.Conv2d):        
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)


def logger(exp_path, exp_name, work_dir, exception, resume=False):

    from tensorboardX import SummaryWriter
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if not resume:
        copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)


    return writer, log_file



def logger_for_CMTL(exp_path, exp_name, work_dir, exception, resume=False):
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    if not os.path.exists(exp_path+ '/' + exp_name):
        os.mkdir(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if not resume:
        copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)


    return log_file

def logger_txt(log_file,epoch,scores):

    mae, mse, loss = scores

    snapshot_name = 'all_ep_%d_mae_%.1f_mse_%.1f' % (epoch + 1, mae, mse)

    # pdb.set_trace()

    with open(log_file, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')
        f.write(snapshot_name + '\n')
        f.write('    [mae %.2f mse %.2f], [val loss %.4f]\n' % (mae, mse, loss))
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')    





def vis_results(exp_name, epoch, writer, restore, img, pred_map, gt_map):

    pil_to_tensor = standard_transforms.ToTensor()

    x = []
    
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map)):
        if idx>1:# show only one group
            break
        pil_input = restore(tensor[0])
        pil_output = torch.from_numpy(tensor[1]/(tensor[2].max()+1e-10)).repeat(3,1,1)
        pil_label = torch.from_numpy(tensor[2]/(tensor[2].max()+1e-10)).repeat(3,1,1)
        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_label, pil_output])
    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy()*255).astype(np.uint8)

    writer.add_image(exp_name + '_epoch_' + str(epoch+1), x)



def print_summary(exp_name,scores,train_record):
    mae, mse, loss = scores
    print( '='*50 )
    print( exp_name )
    print( '    '+ '-'*20 )
    print( '    [mae %.2f mse %.2f], [val loss %.4f]' % (mae, mse, loss) )        
    print( '    '+ '-'*20 )
    print( '[best] [model: %s] , [mae %.2f], [mse %.2f]' % (train_record['best_model_name'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse']) )
    print( '='*50)

def print_WE_summary(log_txt,epoch,scores,train_record,c_maes):
    mae, mse, loss = scores
    # pdb.set_trace()
    with open(log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(epoch) + '\n\n')
        f.write('  [mae %.4f], [val loss %.4f]\n\n' % (mae, loss))
        f.write('    list: ' + str(np.transpose(c_maes.avg)) + '\n')


        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( '    '+ '-'*20 )
    print( '    [mae %.2f mse %.2f], [val loss %.4f]' % (mae, mse, loss) )        
    print( '    '+ '-'*20 )
    print( '[best] [model: %s] , [mae %.2f], [mse %.2f]' % (train_record['best_model_name'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse']) )
    print( '='*50 )


def print_GCC_summary(log_txt,epoch, scores,train_record,c_maes,c_mses):
    mae, mse, loss = scores
    c_mses['level'] = np.sqrt(c_mses['level'].avg)
    c_mses['time'] = np.sqrt(c_mses['time'].avg)
    c_mses['weather'] = np.sqrt(c_mses['weather'].avg)
    with open(log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(epoch) + '\n\n')
        f.write('  [mae %.4f mse %.4f], [val loss %.4f]\n\n' % (mae, mse, loss))
        f.write('  [level: mae %.4f mse %.4f]\n' % (np.average(c_maes['level'].avg), np.average(c_mses['level'])))
        f.write('    list: ' + str(np.transpose(c_maes['level'].avg)) + '\n')
        f.write('    list: ' + str(np.transpose(c_mses['level'])) + '\n\n')

        f.write('  [time: mae %.4f mse %.4f]\n' % (np.average(c_maes['time'].avg), np.average(c_mses['time'])))
        f.write('    list: ' + str(np.transpose(c_maes['time'].avg)) + '\n')
        f.write('    list: ' + str(np.transpose(c_mses['time'])) + '\n\n')

        f.write('  [weather: mae %.4f mse %.4f]\n' % (np.average(c_maes['weather'].avg), np.average(c_mses['weather'])))
        f.write('    list: ' + str(np.transpose(c_maes['weather'].avg)) + '\n')
        f.write('    list: ' + str(np.transpose(c_mses['weather']))+ '\n\n')

        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( '    '+ '-'*20 )
    print( '    [mae %.2f mse %.2f], [val loss %.4f]' % (mae, mse, loss) )
    print( '    '+ '-'*20 )
    print( '[best] [model: %s] , [mae %.2f], [mse %.2f]' % (train_record['best_model_name'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse']) )
    print( '='*50 )   


def update_model(net,optimizer,scheduler,epoch,i_tb,exp_path,exp_name,scores,train_record,log_file=None):

    mae, mse, loss = scores

    snapshot_name = 'all_ep_%d_mae_%.1f_mse_%.1f' % (epoch + 1, mae, mse)

    if mae < train_record['best_mae'] or mse < train_record['best_mse']:   
        train_record['best_model_name'] = snapshot_name
        if log_file is not None:
            logger_txt(log_file,epoch,scores)
        to_saved_weight = net.state_dict()
        torch.save(to_saved_weight, os.path.join(exp_path, exp_name, snapshot_name + '.pth'))

    if mae < train_record['best_mae']:           
        train_record['best_mae'] = mae
    if mse < train_record['best_mse']:
        train_record['best_mse'] = mse 

    latest_state = {'train_record':train_record, 'net':net.state_dict(), 'optimizer':optimizer.state_dict(),\
                    'scheduler':scheduler.state_dict(), 'epoch': epoch, 'i_tb':i_tb, 'exp_path':exp_path, \
                    'exp_name':exp_name}

    torch.save(latest_state,os.path.join(exp_path, exp_name, 'latest_state.pth'))

    return train_record


def copy_cur_env(work_dir, dst_dir, exception):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)


        if os.path.isdir(file) and exception not in filename:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count

class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,num_class):        
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, cur_val, class_id):
        self.cur_val[class_id] = cur_val
        self.sum[class_id] += cur_val
        self.count[class_id] += 1
        self.avg[class_id] = self.sum[class_id] / self.count[class_id]


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff








