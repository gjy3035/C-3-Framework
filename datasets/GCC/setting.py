from easydict import EasyDict as edict

# init
__C_SHHA = edict()

cfg_data = __C_SHHA

__C_SHHA.STD_SIZE = (544,960)
__C_SHHA.TRAIN_SIZE = (480,848)
__C_SHHA.DATA_PATH = '/media/D/DataSet/GCC'

__C_SHHA.VAL_MODE = 'rd' # rd: radomn splitting; cc: cross camera; cl: cross location              

__C_SHHA.MEAN_STD = ([0.302234709263, 0.291243076324, 0.269087553024], [0.227743327618, 0.211051672697, 0.184846073389])

__C_SHHA.LABEL_FACTOR = 1
__C_SHHA.LOG_PARA = 100.

__C_SHHA.RESUME_MODEL = ''#model path
__C_SHHA.TRAIN_BATCH_SIZE = 4 #imgs

__C_SHHA.VAL_BATCH_SIZE = 4 #


