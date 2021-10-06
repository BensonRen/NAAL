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

def AL_from_flag(flags):
    """
    AL interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    # Make Network
    ntwk = Network(NA, flags)
    # Active learning
    ntwk.active_learn()

def AL_debug(flags):
    ntwk = Network(NA, flags)
    ntwk.get_training_data_distribution(iteration_ind=-1)
    ntwk.plot_sine_debug_plot(iteration_ind=-1)
    ntwk.train()
    ntwk.get_training_data_distribution(iteration_ind=1)
    ntwk.plot_sine_debug_plot(iteration_ind=1)

def hyper_sweep_AL():
    """
    The code to hyper-sweep the active learning
    """
    for al_mode in ['Random','MSE']:
        for al_n_step in [20]:
            #for al_n_dx in [5, 10, 20]:
            for al_n_dx in [20]:
                for al_n_x0 in [20, 50, 100]:
                    for al_x_pool_factor in [0.001, 0.01, 0.1, 0.2, 0.5]:       # The size of the pool divided by the number of points chosen
                        flags = flag_reader.read_flag()  	#setting the base case
                        flags.al_mode = al_mode
                        flags.al_n_step = al_n_step
                        flags.al_n_dx = al_n_dx
                        flags.al_n_x0 = al_n_x0
                        flags.al_x_pool = int(al_n_dx / al_x_pool_factor)
                        AL_from_flag(flags)

if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()  	#setting the base case
    #AL_from_flag(flags)
    #AL_debug(flags)
    hyper_sweep_AL()