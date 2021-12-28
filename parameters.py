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
AL_TEST_N = 2000
AL_X_POOL = 500
BOOTSTRAP = 0     # Bootstraping the training dataset for each of the models during training
LOAD_DATASET = None
LOAD_TESTSET = None
RESET_WEIGHT = False 
PLOT_COR_VAR_MSE = True
PLOT = True
SHUFFLE_EACH_MODEL = True
PLOT_DIR = 'results/fig/naal_test'

STOP_CRITERIA_NUM = 5

# Dataset related parameters
# DATA_SET = 'sine'
# MSE_CUTOFF = 1e-3
# NA_NUM_INIT = 100
# AL_N_dX = 10       # The number of data points to add at each step
# AL_N_X0 = 40      # The starting size of the dataset
# DIM_X = 1
# DIM_Y = 1
# LINEAR = [DIM_X, 20,  20, 20, 20, 20, 20, 20, 20, 20, DIM_Y]

DATA_SET = 'robo'
MSE_CUTOFF = 1e-3
NA_NUM_INIT = 200
AL_N_dX = 20       # The number of data points to add at each step
AL_N_X0 = 1000       # The starting size of the dataset
DIM_X = 4
DIM_Y = 2
LINEAR = [DIM_X, 500, 500, 500, 500, DIM_Y]

# DATA_SET = 'meta'
# MSE_CUTOFF = 1e-3
# AL_N_dX = 100       # The number of data points to add at each step
# AL_N_X0 = 1000       # The starting size of the dataset
# DIM_X = 5
# DIM_Y = 201

DIM_X_LOW = [-1]
DIM_X_HIGH = [1]
# FREQ = 30

# Model Architectural Params 

# Optimizer Params
OPTIM = "Adam"
REG_SCALE = 1e-4
BATCH_SIZE = 2096
EVAL_STEP = 500
TRAIN_STEP = 1000
LEARN_RATE = 1e-3
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-9

# Data specific Params
MODEL_NAME = None 
DATA_DIR = 'data'                                               # All simulated simple dataset
