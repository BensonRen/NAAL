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

if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()  	#setting the base case
    AL_from_flag(flags)
