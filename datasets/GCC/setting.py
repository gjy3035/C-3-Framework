from easydict import EasyDict as edict

# init
__C_GCC = edict()

cfg_data = __C_GCC

__C_GCC.STD_SIZE = (544,960)
__C_GCC.TRAIN_SIZE = (272,480)
__C_GCC.DATA_PATH = '../ProcessedData/GCC'

__C_GCC.VAL_MODE = 'cc' # rd: radomn splitting; cc: cross camera; cl: cross location

__C_GCC.DATA_GT = 'k15_s4'            

__C_GCC.MEAN_STD = ([0.302234709263, 0.291243076324, 0.269087553024], [0.227743327618, 0.211051672697, 0.184846073389])

__C_GCC.LABEL_FACTOR = 1
__C_GCC.LOG_PARA = 1000.

__C_GCC.RESUME_MODEL = ''#model path
__C_GCC.TRAIN_BATCH_SIZE = 16 #imgs

__C_GCC.VAL_BATCH_SIZE = 16 #


