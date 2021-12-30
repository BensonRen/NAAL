"""
Params for Back propagation model
"""
# Define which data set you are using
from math import inf

NAAL = True
# Active Learning related parameters
AL_N_STEP = 1      # Total number of steps of the datapoint
AL_N_STEP_CAP = 300      # Total number of steps of the datapoint
AL_N_MODEL = 5     # The number of models
AL_MODE = 'Random'     # The Active Learning mode, MSE means using the true forward error
# AL_MODE = 'NA'     # The Active Learning mode, MSE means using the true forward error
NAAL_STEPS = 100
NA_MD_RADIUS = 0.05
# NA_MD_COEFF = 100

#AL_MODE = 'VAR'     # The Active Learning mode, VAE means using the variance
AL_TEST_N = 4000
AL_X_POOL = 500
BOOTSTRAP = 0     # Bootstraping the training dataset for each of the models during training
LOAD_DATASET = None
LOAD_TESTSET = 'saved_datasets'
RESET_WEIGHT = False 
PLOT_COR_VAR_MSE = True
PLOT = True
SHUFFLE_EACH_MODEL = True
PLOT_DIR = 'results/fig/naal_test'

STOP_CRITERIA_NUM = 5

CONV_OUT_CHANNEL = []
CONV_KERNEL_SIZE = []
CONV_STRIDE = []

# Dataset related parameters
DATA_SET = 'sine'
MSE_CUTOFF = 5e-4
NA_NUM_INIT = 100
AL_N_dX = 10       # The number of data points to add at each step
AL_N_X0 = 40      # The starting size of the dataset
DIM_X = 1
DIM_Y = 1
LINEAR = [DIM_X, 20,  20, 20, 20, 20, 20, 20, 20, 20, DIM_Y]
if AL_MODE == 'Dropout':
    LINEAR = [DIM_X, 60,60,60,60,60,60,60,60,60,  DIM_Y]
REG_SCALE = 1e-4
LR_DECAY_RATE = 0.5

# DATA_SET = 'robo'
# MSE_CUTOFF = 1e-4
# NA_NUM_INIT = 100
# AL_N_dX = 10       # The number of data points to add at each step
# AL_N_X0 = 40       # The starting size of the dataset
# DIM_X = 4
# DIM_Y = 2
# LINEAR = [DIM_X, 500, 500, 500, 500, DIM_Y]
# if AL_MODE == 'Dropout':
#     LINEAR = [DIM_X, 1500, 1500, 1500, 1500, DIM_Y]
    

# DATA_SET = 'ADM'
# MSE_CUTOFF = 2e-3
# NA_NUM_INIT = 1000
# AL_N_dX = 100       # The number of data points to add at each step
# AL_N_X0 = 500       # The starting size of the dataset
# DIM_X = 14
# DIM_Y = 2000
# LINEAR = [DIM_X, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1000]
# CONV_OUT_CHANNEL = [4, 4, 4]
# CONV_KERNEL_SIZE = [4, 3, 3]
# CONV_STRIDE = [2, 1, 1]
# if AL_MODE == 'Dropout':
#     LINEAR = [DIM_X, 1500, 1500, 1500, 1500, DIM_Y]
# REG_SCALE = 1e-7


# DATA_SET = 'Shell'
# MSE_CUTOFF = 2e-3
# NA_NUM_INIT = 200
# AL_N_dX = 20       # The number of data points to add at each step
# AL_N_X0 = 40       # The starting size of the dataset
# DIM_X = 8
# DIM_Y = 201
# LINEAR = [DIM_X, 100, 100, DIM_Y]
# CONV_OUT_CHANNEL = []
# CONV_KERNEL_SIZE = []
# CONV_STRIDE = []
# if AL_MODE == 'Dropout':
#     LINEAR = [DIM_X, DIM_Y]
# REG_SCALE = 0
# LR_DECAY_RATE = 0.8


# DATA_SET = 'Stack'
# MSE_CUTOFF = 2e-5
# NA_NUM_INIT = 200
# AL_N_dX = 20       # The number of data points to add at each step
# AL_N_X0 = 50       # The starting size of the dataset
# DIM_X = 5
# DIM_Y = 256
# LINEAR = [DIM_X, 700, 700, 700, 700, 700, 700, 700, 700, 700,  DIM_Y]
# if AL_MODE == 'Dropout':
#     LINEAR = [DIM_X, 1500, 1500, 1500, 1500, DIM_Y]
# REG_SCALE = 0
# LR_DECAY_RATE = 0.8

DIM_X_LOW = [-1]
DIM_X_HIGH = [1]

# Model Architectural Params 

# Optimizer Params
OPTIM = "Adam"
BATCH_SIZE = 2096
EVAL_STEP = 500
TRAIN_STEP = 1000
LEARN_RATE = 1e-3
LR_DECAY_RATE = 0.8
STOP_THRESHOLD = 1e-9

# Data specific Params
MODEL_NAME = None 
DATA_DIR = 'data'                                               # All simulated simple dataset
