import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035
__C.DATASET = 'SHHA' # SHHA, SHHB, UCF50, UCFQNRF, WE

if __C.DATASET == 'UCF50':# only for UCF50
	from datasets.UCF50.setting import cfg_data
	__C.VAL_INDEX = cfg_data.VAL_INDEX 

if __C.DATASET == 'GCC':# only for GCC
	from datasets.GCC.setting import cfg_data
	__C.VAL_MODE = cfg_data.VAL_MODE 


__C.NET = 'MCNN'

__C.PRE_GCC = False
__C.PRE_GCC_MODEL = ''

__C.GPU_ID = [0]

# learning rate settings
__C.LR = 1e-4
__C.LR_DECAY = 1
__C.LR_DECAY_START = -1
__C.NUM_EPOCH_LR_DECAY = 1 # epoches
__C.MAX_EPOCH = 1000

# print 
__C.PRINT_FREQ = 5

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
			 + '_adam_' + __C.DATASET \
             + '_' + __C.NET \
             + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':
	__C.EXP_NAME += '_' + str(__C.VAL_INDEX)	

if __C.DATASET == 'GCC':
	__C.EXP_NAME += '_' + __C.VAL_MODE	

__C.EXP_PATH = './exp'

#------------------------------VAL------------------------
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 1 # After 300 epoches, the freq is set as 1

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for SHHA



#================================================================================
#================================================================================
#================================================================================  