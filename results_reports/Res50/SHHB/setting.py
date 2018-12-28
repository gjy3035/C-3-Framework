from easydict import EasyDict as edict

# init
__C_SHHB = edict()

cfg_data = __C_SHHB

__C_SHHB.STD_SIZE = (768,1024)
__C_SHHB.TRAIN_SIZE = (576,768)
__C_SHHB.DATA_PATH = '/media/D/DataSet/CC/768x1024RGB-k15-s4/shanghaitech_part_B'               

__C_SHHB.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

__C_SHHB.LABEL_FACTOR = 1
__C_SHHB.LOG_PARA = 100.

__C_SHHB.RESUME_MODEL = ''#model path
__C_SHHB.TRAIN_BATCH_SIZE = 6 #imgs

__C_SHHB.VAL_BATCH_SIZE = 1 # must be 1


