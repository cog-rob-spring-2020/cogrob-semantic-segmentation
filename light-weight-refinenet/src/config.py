import numpy as np

# DATASET PARAMETERS
TRAIN_DIR = '/opt/carla/PythonAPI/carla_scripts/training_data'
VAL_DIR = TRAIN_DIR
#TRAIN_DIR = '/opt/carla/PythonAPI/carla_scripts/light-weight-refinenet/data'
#VAL_DIR = TRAIN_DIR

#TRAIN_LIST = ['/opt/carla/PythonAPI/carla_scripts/training_data/test_train.nyu']
#VAL_LIST = ['/opt/carla/PythonAPI/carla_scripts/training_data/test_val.nyu']

TRAIN_LIST = ['/opt/carla/PythonAPI/carla_scripts/training_data/train.nyu'] * 3 
VAL_LIST = ['/opt/carla/PythonAPI/carla_scripts/training_data/val.nyu'] * 3
#TRAIN_LIST = ['/opt/carla/PythonAPI/carla_scripts/light-weight-refinenet/data/train.nyu']
#VAL_LIST = ['/opt/carla/PythonAPI/carla_scripts/light-weight-refinenet/data/val.nyu']

SHORTER_SIDE = [350] * 3
CROP_SIZE = [500] * 3

NORMALISE_PARAMS = [1./255, # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)), # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))] # STD
BATCH_SIZE = [1] * 3
NUM_WORKERS = 16
NUM_CLASSES = [255] * 3
LOW_SCALE = [0.5] * 3
HIGH_SCALE = [2.0] * 3
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = '50'
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
EVALUATE = False
FREEZE_BN = [True] * 3
NUM_SEGM_EPOCHS = [100] * 3
PRINT_EVERY = 10
RANDOM_SEED = 42
SNAPSHOT_DIR = './ckpt/'
CKPT_PATH = './ckpt/checkpoint.pth.tar'
VAL_EVERY = [5] * 3 # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR_ENC = [5e-4, 2.5e-4, 1e-4]  # TO FREEZE, PUT 0
LR_DEC = [5e-3, 2.5e-3, 1e-3]
MOM_ENC = [0.9] * 3 # TO FREEZE, PUT 0
MOM_DEC = [0.9] * 3
WD_ENC = [1e-5] * 3 # TO FREEZE, PUT 0
WD_DEC = [1e-5] * 3
OPTIM_DEC = 'sgd'
