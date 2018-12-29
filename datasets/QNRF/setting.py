from easydict import EasyDict as edict

# init
__C_QNRF = edict()

cfg_data = __C_QNRF

__C_QNRF.STD_SIZE = (768,1024)
__C_QNRF.TRAIN_SIZE = (576,768)
__C_QNRF.DATA_PATH = '../ProcessedData/UCF-qnrf/1024x1024_mod16'               

__C_QNRF.MEAN_STD = ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449]) 

__C_QNRF.LABEL_FACTOR = 1
__C_QNRF.LOG_PARA = 100.

__C_QNRF.RESUME_MODEL = ''#model path
__C_QNRF.TRAIN_BATCH_SIZE = 1 #imgs

__C_QNRF.VAL_BATCH_SIZE = 1 #


