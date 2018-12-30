from easydict import EasyDict as edict

# init
__C_SHHA = edict()

cfg_data = __C_SHHA

__C_SHHA.STD_SIZE = (768,1024)
__C_SHHA.TRAIN_SIZE = (576,768)
__C_SHHA.DATA_PATH = '../ProcessedData/Shanghai_proA'               

__C_SHHA.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])

__C_SHHA.LABEL_FACTOR = 1
__C_SHHA.LOG_PARA = 100.

__C_SHHA.RESUME_MODEL = ''#model path
__C_SHHA.TRAIN_BATCH_SIZE = 1 #imgs

__C_SHHA.VAL_BATCH_SIZE = 1 #


