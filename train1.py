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
    model_fn = NAAL if flags.naal else NN
    model_fn = Dropout_model if 'Drop' in flags.al_mode else NAAL         # This is the model for dropout
    # Make Network
    ntwk = Network(model_fn, flags)
    # Active learning
    ntwk.active_learn(trail=trail)

def AL_debug(flags):
    model_fn = NAAL if flags.naal else NN
    model_fn = Dropout_model if 'Drop' in flags.al_mode else NAAL
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
    # for reset_weight in [True, False]:
    for reset_weight in [False]:
    # for reset_weight in [False]:
        # for al_mode in ['Random']:
        # for al_mode in ['NA']:
        # for al_mode in ['VAR']:
        # for al_mode in ['NAMD_POW']:
        # for al_mode in ['Dropout']:
        # for al_mode in ['Core-set']:
        # for al_mode in ['MSE']:
        # for al_mode in ['Random','NAMD_POW','VAR']:
            # for al_n_step in [-1]:
            #for al_n_step in [20]:
                # for al_n_dx in [10]:
                #for al_n_dx in [1, 5, 10, 20, 50]:
                    # for al_n_x0 in [20]:
                    #for al_n_x0 in [20, 50, 100, 200]:
            # for bs in [200]:
            # for bs in [500, 1000, 2000]:
            for al_x_pool_factor in [0.2]:#, 0.1, 0.2, 0.25]:       # The size of the pool divided by the number of points chosen
            #for al_x_pool_factor in [0.25]:       # The size of the pool divided by the number of points chosen
            # for al_x_pool_factor in [0.1]:       # The size of the pool divided by the number of points chosen
            #for al_x_pool_factor in [0.05]:       # The size of the pool divided by the number of points chosen
            #for al_x_pool_factor in [0.5, 0.1, 0.05]:       # The size of the pool divided by the number of points chosen
                for n_models in [10]:
                    # ii = 0
                    # for i in range(ii, ii+1):                                      # Total number of trails to aggregate
                    # for i in range(10):                                      # Total number of trails to aggregate
                    for i in range(5):                                      # Total number of trails to aggregate
                    # for i in range(5, 10):                                      # Total number of trails to aggregate
                        flags = flag_reader.read_flag()  	#setting the base case
                        # flags.batch_size = 2048
                        # flags.train_step = 100
                        # flags.batch_size = bs
                        flags.train_step = 300
                        flags.reset_weight = reset_weight
                        flags.al_n_model = n_models
                        flags.al_mode = al_mode
                        # Get the actual al_step
                        # if al_n_step == -1:
                        #     flags.al_n_step = (num_train_upper - al_n_x0) // al_n_dx
                        # else:
                        #     flags.al_n_step = al_n_step
                        # flags.al_n_dx = al_n_dx
                        # flags.al_n_x0 = al_n_x0
                        if al_x_pool_factor is None:
                            al_x_pool_factor = 0.2      # 5 times the pool
                        flags.al_x_pool = int(flags.al_n_dx / al_x_pool_factor)
                        num_neuron_per_layer = flags.linear[1]      # This is marked here since Dropout might change and make new folder
                        # If dropout, we need to do the triple for the compensation
                        if flags.al_mode == 'Dropout':
                            flags.train_step = 1000
                            flags.lr = 0.001
                            flags.eval_step = 100
                            flags.batch_size= 20
                            for kk in range(1, len(flags.linear)-1):
                                # flags.linear[i] *= int(2)
                                flags.linear[kk] *= int(math.sqrt(flags.al_n_model)) # 10 represents the ensemble of 10 models
                        # Set the plotting directory
                        #flags.plot_dir = 'results/correlation_trail'
                        #flags.plot_dir = 'results/prior_test'
                        #flags.plot_dir = 'results/30_sine_smaller_model_dx_1'
                        #flags.plot_dir = 'results/30_sine_nmod_20_shuffle_to500_dx_10_pool_mul_10'
                        #flags.plot_dir = 'results/30_sine_num_layer_{}_nmod_{}_to{}_dx_{}_pool_mul_{}'.format(len(flags.linear) - 2, n_models, num_train_upper, al_n_dx, int(1/al_x_pool_factor))
                        flags.plot_dir = 'results/{}_num_layer_{}_nuron_{}_nmod_{}_toMSE_{}_dx_{}_pool_mul_{}_naal_{}'.format(flags.data_set, len(flags.linear) - 2, 
                                num_neuron_per_layer, n_models, flags.mse_cutoff, flags.al_n_dx, int(1/al_x_pool_factor), flags.naal)
                        #flags.plot_dir = 'results/30_sine_nmod_20_bootstrap_0.9'
                        #flags.plot_dir = 'results/30_sine_nmod_5_trail_5'
                        #flags.plot_dir = 'results/30_sine_nmod_20_add_noise_add_2'
                        #flags.plot_dir = 'results/single_model_30_sine'
                        #flags.plot_dir = 'results/testing_random_state'
                        
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
