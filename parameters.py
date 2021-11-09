"""
Params for Back propagation model
"""
# Define which data set you are using
from math import inf

NAAL = False
# Active Learning related parameters
AL_N_STEP = 1      # Total number of steps of the datapoint
AL_N_dX = 100       # The number of data points to add at each step
AL_N_X0 = 1000       # The starting size of the dataset
AL_N_MODEL = 5     # The number of models
AL_MODE = 'Random'     # The Active Learning mode, MSE means using the true forward error
# AL_MODE = 'NA'     # The Active Learning mode, MSE means using the true forward error
NAAL_STEPS = 20
#AL_MODE = 'VAR'     # The Active Learning mode, VAE means using the variance
AL_TEST_N = 1000
AL_X_POOL = 500
BOOTSTRAP = 0     # Bootstraping the training dataset for each of the models during training
LOAD_DATASET = None
LOAD_TESTSET = None
RESET_WEIGHT = False 
PLOT_COR_VAR_MSE = True
PLOT = True
PLOT_DIR = 'results/fig/naal_test'

# Dataset related parameters
#DATA_SET = 'sine'
DATA_SET = 'xsinx'
DIM_X = 1
DIM_Y = 1
DIM_X_LOW = [-1]
DIM_X_HIGH = [1]
FREQ = 30
# DIM_X_LOW = [-30]
# DIM_X_HIGH = [30]

# Model Architectural Params 
LINEAR = [DIM_X, 20,  20, 20, 20, 20, 20, 20, DIM_Y]

# Optimizer Params
OPTIM = "Adam"
REG_SCALE = 0
BATCH_SIZE = 500
EVAL_STEP = 50
TRAIN_STEP = 1000
LEARN_RATE = 1e-3
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-9

# Data specific Params
MODEL_NAME = None 
DATA_DIR = 'data'                                               # All simulated simple dataset
