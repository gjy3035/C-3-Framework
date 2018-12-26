from easydict import EasyDict as edict

# init
__C_SHHA = edict()

cfg_data = __C_SHHA

__C_SHHA.STD_SIZE = (768,1024)
__C_SHHA.TRAIN_SIZE = (576,768)
__C_SHHA.DATA_PATH = '/media/D/DataSet/CC/Shanghai_proA'               

__C_SHHA.MEAN_STD = ([0.504379212856, 0.510956227779, 0.505369007587], [0.22513884306, 0.225588873029, 0.22579960525])

__C_SHHA.LABEL_FACTOR = 1
__C_SHHA.LOG_PARA = 100.

__C_SHHA.RESUME_MODEL = ''#model path
__C_SHHA.TRAIN_BATCH_SIZE = 4 #imgs

__C_SHHA.VAL_BATCH_SIZE = 1 #


