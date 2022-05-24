"""
Params for Back propagation model
"""
# Define which data set you are using
from math import inf

NAAL = True
# Active Learning related parameters
AL_N_STEP = 1      # Total number of steps of the datapoint
AL_N_STEP_CAP = 400      # Total number of steps of the datapoint
AL_N_MODEL = 10     # The number of models
AL_MODE = 'Random'     # The Active Learning mode, MSE means using the true forward error
# AL_MODE = 'NA'     # The Active Learning mode, MSE means using the true forward error
NAAL_STEPS = 1000
NA_MD_RADIUS = 0.05
# NA_MD_COEFF = 100
#W_DIV = 0.5             # The diversity score lambda value for the diversity
#W_DEN = 0             # The diversity score lambda value for the density

W_DIV = 0.33             # The diversity score lambda value for the diversity
W_DEN = 0.33             # The diversity score lambda value for the density

#AL_MODE = 'VAR'     # The Active Learning mode, VAE means using the variance
AL_TEST_N = 4000
AL_X_POOL = 500
BOOTSTRAP = 0     # Bootstraping the training dataset for each of the models during training
LOAD_DATASET = None
LOAD_TESTSET =  None# 'saved_datasets'
RESET_WEIGHT = False 
PLOT_COR_VAR_MSE = False
PLOT = True
SHUFFLE_EACH_MODEL = True
PLOT_DIR = 'results/fig/naal_test'

STOP_CRITERIA_NUM = 5

CONV_OUT_CHANNEL = []
CONV_KERNEL_SIZE = []
CONV_STRIDE = []

DROPOUT_P = 0.5

# Dataset related parameters
#DATA_SET = 'sine'
#MSE_CUTOFF = 3e-4
#BATCH_SIZE = 5000
#NALR = 0.005
#NA_LR_DECAY_RATE = 0.2
#AL_N_dX = 40       # The number of data points to add at each step
#AL_N_X0 = 80      # The starting size of the dataset
#NA_NUM_INIT = AL_N_dX * 1
#DIM_X = 1
#DIM_Y = 1
#LEARN_RATE = 1e-4
#LINEAR = [DIM_X, 20, 20, 20, 20, 20, 20, 20, 20, 20, DIM_Y]
#REG_SCALE = 0 # 0 for #1e-4
#LR_DECAY_RATE = 0.8

#DATA_SET = 'robo'
#MSE_CUTOFF = 1e-4
#BATCH_SIZE = 5000
#NALR = 0.005
#NA_LR_DECAY_RATE = 0.2
#AL_N_dX = 40       # The number of data points to add at each step
#AL_N_X0 = 80       # The starting size of the dataset
#NA_NUM_INIT = AL_N_dX * 1
#DIM_X = 4
#DIM_Y = 2
#REG_SCALE = 1e-4 # 0 for #1e-4
#LINEAR = [DIM_X, 500, 500, 500, 500, DIM_Y]
#LEARN_RATE = 1e-3
#LR_DECAY_RATE = 0.8
    

#DATA_SET = 'scon'           # Superconductor dataset
#MSE_CUTOFF = 3e-3
#BATCH_SIZE = 5000
#NALR = 0.005
#NA_LR_DECAY_RATE = 0.2
#AL_N_dX = 40       # The number of data points to add at each step
#AL_N_X0 = 80       # The starting size of the dataset
#NA_NUM_INIT = AL_N_dX * 1
#DIM_X = 81
#DIM_Y = 1
#REG_SCALE = 1e-3 # 0 for #1e-4
#LINEAR = [DIM_X, 200, 200, 200, 200, DIM_Y]
#LEARN_RATE = 1e-3
#LR_DECAY_RATE = 0.8

DATA_SET = 'hydro'           # Superconductor dataset
MSE_CUTOFF = 7e-3
BATCH_SIZE = 5000
NALR = 0.005
NA_LR_DECAY_RATE = 0.2
AL_N_dX = 40       # The number of data points to add at each step
AL_N_X0 = 80       # The starting size of the dataset
NA_NUM_INIT = AL_N_dX * 1
DIM_X = 6
DIM_Y = 1
REG_SCALE = 1e-4 # 0 for #1e-4
LINEAR = [DIM_X, 50, 50, 50, 50, 50, 50, DIM_Y]
LEARN_RATE = 1e-3
LR_DECAY_RATE = 0.8


#DATA_SET = 'airfoil'           # Superconductor dataset
#MSE_CUTOFF = 3e-3
#BATCH_SIZE = 5000
#NALR = 0.005
#NA_LR_DECAY_RATE = 0.2
#AL_N_dX = 40       # The number of data points to add at each step
#AL_N_X0 = 80       # The starting size of the dataset
#NA_NUM_INIT = AL_N_dX * 1
#DIM_X = 5
#DIM_Y = 1
#REG_SCALE = 1e-4 # 0 for #1e-4
#LINEAR = [DIM_X, 200, 200, 200, 200, DIM_Y]
#LEARN_RATE = 1e-3
#LR_DECAY_RATE = 0.8

# DATA_SET = 'ADM'
# AL_N_MODEL = 5     # The number of models
# MSE_CUTOFF = 9e-4
# BATCH_SIZE = 500
# NALR = 0.005
# NA_LR_DECAY_RATE = 0.2
# AL_N_dX = 100       # The number of data points to add at each step
# AL_N_X0 = 100       # The starting size of the dataset
# NA_NUM_INIT = AL_N_dX * 1
# DIM_X = 14
# DIM_Y = 2000
# LINEAR = [DIM_X, 1500, 1500, 1500, 1500, 1500, 1500, 1000]
# # LINEAR = [DIM_X, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1000]
# CONV_OUT_CHANNEL = [4, 4, 4]
# CONV_KERNEL_SIZE = [4, 3, 3]
# CONV_STRIDE = [2, 1, 1]
# REG_SCALE = 0
# LEARN_RATE = 5e-3
# LR_DECAY_RATE = 0.8


# DATA_SET = 'Shell'
# MSE_CUTOFF = 2e-3
# BATCH_SIZE = 2000
# NA_NUM_INIT = 200
# AL_N_dX = 20       # The number of data points to add at each step
# AL_N_X0 = 40       # The starting size of the dataset
# DIM_X = 8
# DIM_Y = 201
# LINEAR = [DIM_X, 100, 100, DIM_Y]
# CONV_OUT_CHANNEL = []
# CONV_KERNEL_SIZE = []
# CONV_STRIDE = []
# REG_SCALE = 0
# LR_DECAY_RATE = 0.8


#DATA_SET = 'Stack'
#MSE_CUTOFF = 2e-5
#BATCH_SIZE = 2000
#AL_N_dX = 40       # The number of data points to add at each step
#AL_N_X0 = 80       # The starting size of the dataset
#NA_NUM_INIT = AL_N_dX * 1
#NALR = 0.005
#NA_LR_DECAY_RATE = 0.2
#DIM_X = 5
#DIM_Y = 256
#LINEAR = [DIM_X, 700, 700, 700, 700, 700, 700, 700, 700, 700,  DIM_Y]
#CONV_OUT_CHANNEL = []
#CONV_KERNEL_SIZE = []
#CONV_STRIDE = []
#REG_SCALE = 0
#LEARN_RATE = 1e-3
#LR_DECAY_RATE = 0.8

DIM_X_LOW = [-1]
DIM_X_HIGH = [1]

# Model Architectural Params 

# Optimizer Params
OPTIM = "Adam"
EVAL_STEP = 1000
TRAIN_STEP = 500
# LEARN_RATE = 5e-3
# LR_DECAY_RATE = 0.8
STOP_THRESHOLD = 1e-9

# Data specific Params
MODEL_NAME = None 
DATA_DIR = 'data'                                               # All simulated simple dataset
