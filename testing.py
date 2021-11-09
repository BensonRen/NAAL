"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import random
import torch
import sys
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import NA, NAAL
import numpy as np
import time

def AL_from_flag(flags, trail=0):
    """
    AL interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    model_fn = NAAL if flags.naal else NA
    # Make Network
    ntwk = Network(model_fn, flags)
    # Active learning
    ntwk.active_learn(trail=trail)

def AL_debug(flags):
    model_fn = NAAL if flags.naal else NA
    ntwk = Network(model_fn, flags)
    ntwk.get_training_data_distribution(iteration_ind=-1)
    ntwk.plot_sine_debug_plot(iteration_ind=-1)
    ntwk.train()
    ntwk.get_training_data_distribution(iteration_ind=1)
    ntwk.plot_sine_debug_plot(iteration_ind=1)

def hash_random_seed(reset_weight, al_n_step, al_n_dx, al_n_x0, al_x_pool, n_models, trail):
    """
    This hashes the combination of the input hyper-parameter in a unique and retreivable way (hopefully)
    """
    hashed_random_seed = int(str(int(reset_weight)) + str(al_n_step) + str(al_n_dx) + str(al_n_x0) + \
                        str(al_x_pool) + str(n_models) + str(trail))
    print('hashed_random_seed=', hashed_random_seed)
    
    # To make sure random seed is within range [0, 2^32 - 1]
    hashed_random_seed = hashed_random_seed % (2**32)
    return int(hashed_random_seed)

def hyper_sweep_AL():
    """
    The code to hyper-sweep the active learning
    """
    #num_train_upper = 15
    for reset_weight in [True, False]:
        for batch_size in [128]:
        #for batch_size in [32, 64, 128, 512, 1024]:
            flags = flag_reader.read_flag()  	#setting the base case
            flags.batch_size = batch_size
            flags.reset_weight = reset_weight
            flags.hashed_random_seed = hash_random_seed(flags.reset_weight, flags.al_n_step, flags.al_n_dx, flags.al_n_x0, flags.al_x_pool, flags.al_n_model, trail=0)
            np.random.seed(flags.hashed_random_seed)
            AL_from_flag(flags)
            
                            
if __name__ == '__main__':
    # Read the parameters to be set
    start = time.time()
    flags = flag_reader.read_flag()  	#setting the base case
    #hyper_sweep_AL()
    np.random.seed(0)
    torch.manual_seed(0)
    flags.batch_size = 32
    flags.data_set = 'xsinx'
    flags.eval_step = 1
    flags.al_n_step = 1
    flags.al_n_x0 = 100
    flags.al_n_model = 2
    AL_from_flag(flags)
    #AL_debug(flags)
    print('time = {}s'.format(time.time() - start))
