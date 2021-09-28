"""
This file serves to hold helper functions that is related to the "Flag" object which contains
all the parameters during training and inference
"""
# Built-in
import argparse
# Libs

# Own module
from parameters import *

# Torch

def read_flag():
    """
    This function is to write the read the flags from a parameter file and put them in formats
    :return: flags: a struct where all the input params are stored
    """
    parser = argparse.ArgumentParser()
    # Active Learning related parameters
    parser.add_argument('--al-n-step', default=AL_N_STEP, type=int,  help=' #of steps of the datapoint')
    parser.add_argument('--al-n-dx', default=AL_N_dX, type=int,  help='# of data points to add at each step')
    parser.add_argument('--al-n-x0', default=AL_N_X0, type=int,  help='The starting size of the dataset')
    parser.add_argument('--al-n-model', default=AL_N_MODEL, type=int,  help='# of models')
    parser.add_argument('--al-mode', default=AL_MODE, type=str,  help='The Active Learning mode, MSE means using the true forward error')
    parser.add_argument('--load-dataset', default=LOAD_DATASET, type=str,  help='Default None, if yes then load the dataset from that folder')
    
    # Data_Set parameter
    parser.add_argument('--data-set', default=DATA_SET, type=str, help='which data set you are chosing')
    parser.add_argument('--dim-x', default=DIM_X, type=int, help='Dim of X')
    parser.add_argument('--dim-y', default=DIM_Y, type=int, help='Dim of Y')
    parser.add_argument('--dim-x-low', default=DIM_X_LOW, type=list, help='The list of lower bound of x')
    parser.add_argument('--dim-x-high', default=DIM_X_HIGH, type=list, help='The list of higher bound of x')

    # Model Architectural Params
    parser.add_argument('--linear', type=list, default=LINEAR, help='The fc layers units')

    # Optimizer Params
    parser.add_argument('--optim', default=OPTIM, type=str, help='the type of optimizer that you want to use')
    parser.add_argument('--reg-scale', default=REG_SCALE, type=float,  help='#scale for regularization of dense layers')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataSet')
    parser.add_argument('--lr', default=LEARN_RATE, type=float, help='learning rate')
    parser.add_argument('--lr-decay-rate', default=LR_DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--stop_threshold', default=STOP_THRESHOLD, type=float,
                        help='The threshold below which training should stop')
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='name of the model')
    parser.add_argument('--data-dir', default=DATA_DIR, type=str, help='data directory')
    flags = parser.parse_args()  # This is for command line version of the code
    # flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code
    # flagsVar = vars(flags)
    return flags
