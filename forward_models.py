###
# This is the forward model script 
###
import numpy as np
import os 
import pandas as pd
import joblib

from meta.Stack.generate_chen import multi_process_simulate as chen_sim
from meta.Shell.generate_Peurifoy import multi_process_simulate as Peurifoy_sim
import ADMpredict

def simulator(dataset, x):
    if 'sin' in dataset:
        # Sine wave dataset:  1D -> 1D
        y = x * np.sin(3 * np.sin(x * 30))       #  1D Dataset, Y = x*sin(3*sine(30*x))
        assert np.shape(y)[1] == 1, 'Your dimension of y should be 1 in axis 1'
        return y
    elif 'rob' in dataset:
        # Robotic dataset:   4D -> 2D
        assert np.shape(x)[1] == 4, 'Your dimension of x input in the simulator, \
                robotic dataset is wrong!! currently it has shape {}'.format(np.shape(x))
        arm_len = [0.5, 0.5, 1]
        y = np.zeros([len(x), 2])   # Initialize the y dimensional matrix
        y[:, 1]  = x[:, 0]          # The first dimension of the x is the original y value
        for i in range(3):          # For each of the angles, the robotic arm would have angle x
            y[:, 0] += np.cos(x[:, i+1]*np.pi/2) * arm_len[i]
            y[:, 1] += np.sin(x[:, i+1]*np.pi/2) * arm_len[i]
        return y
    elif 'ADM' in dataset:
        # print('calling the ADM simulator now')
        # Meta-material design dataset:    14D  ->  2000D
        assert np.shape(x)[1] == 14, 'Your dimension of x input in the simualtor, meta material dataset is wrong!!'
        return ADMpredict.ensemble_predict_master('meta/ADM/state_dicts/', x)
    elif 'Stack' in dataset:
        # print('calling the Stack simulator now')
        # Meta-material design dataset:     5D  ->  201D
        x = x*22.5+27.5         # This is reverting the normalization
        return chen_sim(x)
    elif 'Shell' in dataset:
        # print('callilng the Shell simulator')
        x = x * 20. + 50 # Reverting the normalization
        # for i in range(len(x)):
        Ypred = Peurifoy_sim(x)
        # Ypred.append(spec)
        return Ypred
    elif 'scon' in dataset:
        # superconductor dataset
        rf = joblib.load("meta/Super_conductor/super_conductor_oracle_rf.joblib")
        assert np.shape(x)[1] == 81, 'Your dimension of x input in the simualtor, Superconductor dataset is wrong!!'
        return np.expand_dims(rf.predict(x), axis=1)
    elif 'airfoil' in dataset:
        # superconductor dataset
        rf = joblib.load("meta/airfoil/airfoil_oracle_rf.joblib")
        assert np.shape(x)[1] == 5, 'Your dimension of x input in the simualtor, Superconductor dataset is wrong!!'
        return np.expand_dims(rf.predict(x), axis=1)
    elif 'hydro' in dataset:
        # superconductor dataset
        rf = joblib.load("meta/hydro/hydro_oracle_rf.joblib")
        assert np.shape(x)[1] == 6, 'Your dimension of x input in the simualtor, Superconductor dataset is wrong!!'
        return np.expand_dims(rf.predict(x), axis=1)
    else:
        assert dataset == 'None', 'Your dataset name is incorrect, check the flag input!'
# Testing function that plots the output of the 
