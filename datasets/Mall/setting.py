from easydict import EasyDict as edict

# init
__C_MALL = edict()

cfg_data = __C_MALL

__C_MALL.STD_SIZE = (480,640)
__C_MALL.TRAIN_SIZE = (480,640)
__C_MALL.DATA_PATH = '/media/D/DataSet/CC/Mall_k31_s8'               

__C_MALL.MEAN_STD = ([0.537967503071, 0.460666239262, 0.41356408596],[0.220573320985, 0.218155637383, 0.20540446043])

__C_MALL.LABEL_FACTOR = 1
__C_MALL.LOG_PARA = 100.

__C_MALL.RESUME_MODEL = ''#model path
__C_MALL.TRAIN_BATCH_SIZE = 4 #imgs

__C_MALL.VAL_BATCH_SIZE = 4 #


