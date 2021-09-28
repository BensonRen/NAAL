"""
Params for Back propagation model
"""
# Define which data set you are using
from math import inf

# Active Learning related parameters
AL_N_STEP = 20      # Total number of steps of the datapoint
AL_N_dX = 100       # The number of data points to add at each step
AL_N_X0 = 100       # The starting size of the dataset
AL_N_MODEL = 5      # The number of models
AL_MODE = 'MSE'     # The Active Learning mode, MSE means using the true forward error
# AL_MODE = 'VAR'     # The Active Learning mode, VAE means using the variance
LOAD_DATASET = None

# Dataset related parameters
DATA_SET = 'sine'
DIM_X = 1
DIM_Y = 1
DIM_X_LOW = [-1]
DIM_X_HIGH = [1]

# Model Architectural Params 
LINEAR = [DIM_X, 20, 20, 20, 20, DIM_Y]

# Optimizer Params
OPTIM = "Adam"
REG_SCALE = 5e-3
BATCH_SIZE = 1024
EVAL_STEP = 20
TRAIN_STEP = 300
LEARN_RATE = 1e-3
LR_DECAY_RATE = 0.8
STOP_THRESHOLD = 1e-9

# Data specific Params
MODEL_NAME = None 
DATA_DIR = 'data'                                               # All simulated simple dataset