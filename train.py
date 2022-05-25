"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
from pickle import NONE
import shutil
import random
import sys

from numpy.lib import math
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import NN, NAAL, Dropout_model
import numpy as np


def AL_from_flag(flags, trail=0):
    """
    AL interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    # if 'Drop' in flags.al_mode:
    #     model_fn = Dropout_model
    # elif flags.nall:
    #     model_fn = NAAL
    # else:
    #     model_fn = NN
    model_fn = NAAL if flags.naal else NN                                   # Use the NAAL model for all, dropout is auxiliary model
    # model_fn = Dropout_model if 'Drop' in flags.al_mode else NAAL         # This is the model for dropout
    # Make Network
    ntwk = Network(model_fn, flags)
    # Active learning
    ntwk.active_learn(trail=trail)

def AL_debug(flags):
    model_fn = NAAL if flags.naal else NN
    # model_fn = Dropout_model if 'Drop' in flags.al_mode else NAAL
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
    num_train_upper = 2000
    al_x_pool_factor = None
    reset_weight = False
    # for reset_weight in [False]:
    #for al_mode in ['Random', 'NA']:
    #for al_mode in ['NA']:
    #for al_mode in ['VAR-Core']:
    #for al_mode in ['VAR']:
    #for al_mode in ['VAR_Div_Den']:            # The Diversity only one, naming historically
    #for al_mode in ['VAR_Div_Den_0.3_0.3']:    # The Diversity and density one
    #for al_mode in ['Random']:
    #for al_mode in ['NAMD_POW']:
    #for al_mode in ['NA_Core']:
    # for al_mode in ['Dropout']:
    #for al_mode in ['Core-set']:
    #for al_mode in ['MSE']:
    for al_mode in ['Random','NA','Core-set','VAR','VAR_Div_Den','VAR_Div_Den_0.3_0.3','Dropout']:
        for al_x_pool_factor in [1/2, 1/4, 1/8, 1/16, 1/32, 1/64]:       # The size of the pool divided by the number of points chosen
            for na_decay in [0.1]:
                for i in range(5):                                      # Total number of trails to aggregate
                    flags = flag_reader.read_flag()  	#setting the base case
                    flags.reset_weight = reset_weight
                    # flags.al_n_model = n_models
                    flags.al_mode = al_mode
                    if al_x_pool_factor is None:
                        al_x_pool_factor = 0.2      # 5 times the pool
                    flags.al_x_pool = int(flags.al_n_dx / al_x_pool_factor)
                    # For NA, we are also doing the same initialization as the pool ratio
                    flags.na_num_init = flags.al_x_pool
                    num_neuron_per_layer = flags.linear[1]      # This is marked here since Dropout might change and make new folder
                    flags.plot_dir = 'results/{}_num_layer_{}_nuron_{}_nmod_{}_toMSE_{}_dx_{}_naal_{}_reg_{}_lr_{}_decay_{}'.format(flags.data_set, len(flags.linear) - 2, 
                                    num_neuron_per_layer, flags.al_n_model, flags.mse_cutoff, flags.al_n_dx, flags.naal, 
                                    flags.reg_scale, flags.lr, flags.lr_decay_rate)
                    
                    
                    # Fix the same random number generator state for the same experiments
                    flags.hashed_random_seed = hash_random_seed(flags.reset_weight, flags.al_n_step, flags.al_n_dx, flags.al_n_x0, flags.al_x_pool, flags.al_n_model, trail=i)
                    np.random.seed(flags.hashed_random_seed)

                    AL_from_flag(flags, trail=i)

if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()  	#setting the base case
    #AL_from_flag(flags)
    #AL_debug(flags)
    hyper_sweep_AL()
