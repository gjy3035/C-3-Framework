from easydict import EasyDict as edict

# init
__C_UCF50 = edict()

cfg_data = __C_UCF50

__C_UCF50.STD_SIZE = (768,1024)
__C_UCF50.TRAIN_SIZE = (576,768)
__C_UCF50.DATA_PATH = '/home/jupyter/crowdcounting/ProcessedData/UCF_CC_50'               

__C_UCF50.MEAN_STD = ([0.403584420681,0.403584420681,0.403584420681], [0.268462955952,0.268462955952,0.268462955952])

__C_UCF50.VAL_INDEX = 1 # 1, 2, 3, 4, 5

__C_UCF50.LABEL_FACTOR = 1
__C_UCF50.LOG_PARA = 100.

__C_UCF50.RESUME_MODEL = ''#model path
__C_UCF50.TRAIN_BATCH_SIZE = 4 #imgs

__C_UCF50.VAL_BATCH_SIZE = 1 #





