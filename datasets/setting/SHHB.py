
from easydict import EasyDict as edict


# init
__C_SHHB = edict()

cfg_data = __C_SHHB

__C_SHHB.TRAIN_SIZE = (512,1024)
# __C_SHHB.TRAIN_SIZE = (256,512)
__C_SHHB.DATA_PATH = '../ProcessedData/part_B_final/'
__C_SHHB.TRAIN_LST = 'train.txt'
__C_SHHB.VAL_LST =  'val.txt'
__C_SHHB.VAL4EVAL = 'val_gt_loc.txt'

__C_SHHB.MEAN_STD = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])

__C_SHHB.LABEL_FACTOR = 1
__C_SHHB.LOG_PARA = 1.

__C_SHHB.RESUME_MODEL = ''#model path
__C_SHHB.TRAIN_BATCH_SIZE = 1 #imgs

__C_SHHB.VAL_BATCH_SIZE = 1 # must be 1


