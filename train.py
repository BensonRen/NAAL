"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import random
import sys
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import NA
import numpy as np

def AL_from_flag(flags, trail=0):
    """
    AL interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    # Make Network
    ntwk = Network(NA, flags)
    # Active learning
    ntwk.active_learn(trail=trail)

def AL_debug(flags):
    ntwk = Network(NA, flags)
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
    #for reset_weight in [True, False]:
    #for reset_weight in [False]:
    for reset_weight in [True]:
        #for al_mode in ['VAR']:
        #for al_mode in ['Random']:
        #for al_mode in ['MSE']:
        #for al_mode in ['VAR','Random','MSE']:
        for al_mode in ['MSE','Random']:
            for al_n_step in [-1]:
            #for al_n_step in [20]:
                for al_n_dx in [50]:
                #for al_n_dx in [1, 5, 10, 20, 50]:
                    for al_n_x0 in [20]:
                    #for al_n_x0 in [20, 50, 100, 200]:
                        for al_x_pool_factor in [0.25, 0.2]:       # The size of the pool divided by the number of points chosen
                        #for al_x_pool_factor in [0.1, 0.02]:       # The size of the pool divided by the number of points chosen
                        #for al_x_pool_factor in [0.01, 0.1, 0.2]:       # The size of the pool divided by the number of points chosen
                            for n_models in [10]:
                                for i in range(10):                                      # Total number of trails to aggregate
                                    flags = flag_reader.read_flag()  	#setting the base case
                                    flags.reset_weight = reset_weight
                                    flags.al_n_model = n_models
                                    flags.al_mode = al_mode
                                    # Get the actual al_step
                                    if al_n_step == -1:
                                        flags.al_n_step = (num_train_upper - al_n_x0) // al_n_dx
                                    else:
                                        flags.al_n_step = al_n_step
                                    flags.al_n_dx = al_n_dx
                                    flags.al_n_x0 = al_n_x0
                                    flags.al_x_pool = int(al_n_dx / al_x_pool_factor)
                                    # Set the plotting directory
                                    #flags.plot_dir = 'results/correlation_trail'
                                    #flags.plot_dir = 'results/prior_test'
                                    #flags.plot_dir = 'results/30_sine_smaller_model_dx_1'
                                    #flags.plot_dir = 'results/30_sine_nmod_20_shuffle_to500_dx_10_pool_mul_10'
                                    flags.plot_dir = 'results/30_sine_nmod_{}_shuffle_to{}_dx_{}_pool_mul_{}'.format(n_models, num_train_upper, al_n_dx, int(1/al_x_pool_factor))
                                    #flags.plot_dir = 'results/30_sine_nmod_20_bootstrap_0.9'
                                    #flags.plot_dir = 'results/30_sine_nmod_5_trail_5'
                                    #flags.plot_dir = 'results/30_sine_nmod_20_add_noise_add_2'
                                    #flags.plot_dir = 'results/single_model_30_sine'
                                    #flags.plot_dir = 'results/testing_random_state'
                                    
                                    # Fix the same random number generator state for the same experiments
                                    #flags.hashed_random_seed = hash_random_seed(flags.reset_weight, flags.al_n_step, flags.al_n_dx, flags.al_n_x0, flags.al_x_pool, flags.al_n_model, trail=i)
                                    #np.random.seed(flags.hashed_random_seed)

                                    AL_from_flag(flags, trail=i)

if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()  	#setting the base case
    #AL_from_flag(flags)
    #AL_debug(flags)
    hyper_sweep_AL()
