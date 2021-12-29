"""
This file serves as a prediction interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
import time
# Torch

# Own
from meta.ADM import flag_reader
from meta.ADM.class_wrapper import Network
from meta.ADM.model_maker import NA
import torch
# Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def load_flags(save_dir, save_file="flags.obj"):
    """
    This function inflate the pickled object to flags object for reuse, typically during evaluation (after training)
    :param save_dir: The place where the obj is located
    :param save_file: The file name of the file, usually flags.obj
    :return: flags
    """
    with open(os.path.join(save_dir, save_file), 'rb') as f:     # Open the file
        flags = pickle.load(f)                                  # Use pickle to inflate the obj back to RAM
    return flags

def predict_from_model(pre_trained_model, X, load_state_dict=None):
    """
    Predicting interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param Xpred_file: The Prediction file position
    :param no_plot: If True, do not plot (For multi_eval)
    :param load_state_dict: The new way to load the model for ensemble MM
    :return: None
    """
    # Retrieve the flag object
    flags = load_flags(pre_trained_model)                       # Get the pre-trained model
    flags.eval_model = pre_trained_model                    # Reset the eval mode
    ntwk = Network(NA, flags)
    # print("Start eval now:")
    
    return ntwk.predict(X, load_state_dict=load_state_dict)

def ensemble_predict(model_list, X):
    """
    This predicts the output from an ensemble of models
    :param model_list: The list of model names to aggregate
    :param Xpred_file: The Xpred_file that you want to predict
    :param model_dir: The directory to plot the plot
    :param no_plot: If True, do not plot (For multi_eval)
    :param remove_extra_files: Remove all the files generated except for the ensemble one
    :param state_dict: New way to load model using state_dict instead of load module
    :return: The prediction Ypred_file
    """
    # print("this is doing ensemble prediction for models :", model_list)
    pred_list = []
    # Get the predictions into a list of np array
    for pre_trained_model in model_list:
        model_folder = 'meta/ADM/model_param'
        pred= predict_from_model(model_folder, X, load_state_dict=pre_trained_model)
        pred_list.append(np.copy(np.expand_dims(pred, axis=2)))
    # Take the mean of the predictions
    pred_all = np.concatenate(pred_list, axis=2)
    pred_mean = np.mean(pred_all, axis=2)
    return pred_mean


def ensemble_predict_master(model_dir, X):
    # print("entering folder to predict:", model_dir)
    model_list = []

    # get the list of models to load
    for model in os.listdir(model_dir):
        # print("entering:", model)
        if 'skip' in model or '.zip' in model :             # For skipping certain folders
            continue;
        if os.path.isdir(os.path.join(model_dir,model)) or '.pt' in model:
            model_list.append(os.path.join(model_dir, model))
    return ensemble_predict(model_list, X)