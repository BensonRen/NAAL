###
# This is the forward model script 
###
import numpy as np
import os 
import pandas as pd


def simulator(dataset, x):
    if 'sin' in dataset:
        # Sine wave dataset:  1D -> 1D
        return x * np.sin(3 * np.sin(x * 30))       #  1D Dataset, Y = x*sin(3*sine(30*x))
    elif 'rob' in dataset:
        # Robotic dataset:   4D -> 2D
        assert np.shape(x)[1] == 4, 'Your dimension of x input in the simulator, robotic dataset is wrong!!'
        arm_len = [0.5, 0.5, 1]
        y = np.zeros([len(x), 2])   # Initialize the y dimensional matrix
        y[:, 1]  = x[:, 0]          # The first dimension of the x is the original y value
        for i in range(3):          # For each of the angles, the robotic arm would have angle x
            y[:, 0] += np.cos(x[:, i+1]) * arm_len[i]
            y[:, 1] += np.sin(x[:, i+1]) * arm_len[i]
        return y
    elif 'meta' in dataset:
        # Meta-material design dataset:    14D  ->  2000D
        assert np.shape(x)[1] == 14, 'Your dimension of x input in the simualtor, meta material dataset is wrong!!'
    else:
        assert dataset == 'None', 'Your dataset name is incorrect, check the flag input!'
# Testing function that plots the output of the 