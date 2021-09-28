"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys
sys.path.append('../utils/')

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

# Libs
import numpy as np
from math import inf
import matplotlib.pyplot as plt
import pandas as pd

class Network(object):
    def __init__(self, model_fn, flags, ckpt_dir=os.path.join(os.path.abspath(''), 'models')):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs    
        if flags.model_name is None:                    # leave custume name if possible
            self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        else:
            self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        
        self.loss = self.make_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.n_model = flags.al_n_model                         # Save the number of models for quick reference

        self.models = []
        # Creating the Committee
        for i in range(flags.al_n_model):
            self.models.append(self.create_model())                        # The model itself
        
        # Setting up the simulator
        self.simulator = self.init_simulator(self.flags.data_set)

        # Creating the dataset
        self.dataset = None
        if flags.load_dataset is None:
            self.data_x, self.data_y = self.init_dataset()
        else:
            self.data_x, self.data_y =  self.load_dataset(flags.load_dataset)
        
    def init_simulator(self, data_set):
        """
        The function to initialize the simulator
        """
        if data_set == 'sine':
            return np.sin

    def scale_uniform_to_data_distribution(self, X):
        """
        since the initialization of the backprop is uniform from [0,1], this function transforms that distribution
        to suitable prior distribution for each dataset. The numbers are accquired from statistics of min and max
        of the X prior given in the training set and data generation process
        :param X: The input uniform distribution from [0,1]
        :return: The transformed initial guess from prior distribution
        """
        X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
        X_new = X * self.build_tensor(X_range) + self.build_tensor(X_lower_bound)
        return X_new
    
    def get_boundary_lower_bound_uper_bound(self):
        """
        Due to the fact that the batched dataset is a random subset of the training set, mean and range would fluctuate.
        Therefore we pre-calculate the mean, lower boundary and upper boundary to avoid that fluctuation. Replace the
        mean and bound of your dataset here
        :return:
        """
        low = self.flags.dim_x_low
        high  = self.flags.dim_x_high
        return np.array(high - low), np.array(low), np.array(high)

    def random_sample_X(self, number):
        """
        The function to randomly (uniformly) sample the X space according to the lower and upper bound
        """
        return self.scale_uniform_to_data_distribution(np.random.uniform(size=[number, self.flags.dim_x]))

    def init_dataset(self):
        """
        The function to initiate and create the dataset
        """
        # Get the initialized x
        data_x = self.random_sample_X(self.flags.al_n_x0)
        data_y = self.simulator(data_x)
        return data_x, data_y
        
    def save_dataset(self, save_name=None):
        """
        The function to save the dataset
        """
        # Make sure the dataset save to somewhere
        if save_name is None:
            save_name = os.path.join('dataset_saved', time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        np.save(os.path.join(save_name, 'data_x.npy'), self.data_x)
        np.save(os.path.join(save_name, 'data_y.npy'), self.data_y)
        print("Your dataset has been saved to ", save_name)

    def load_dataset(self, load_dataset):
        """
        The function to load the dataset
        """
        return np.load(os.path.join(load_dataset, 'data_x.npy')),  np.load(os.path.join(load_dataset, 'data_y.npy'))

    def make_optimizer_eval(self, geometry_eval, optimizer_type=None):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if optimizer_type is None:
            optimizer_type = self.flags.optim
        if optimizer_type == 'Adam':
            op = torch.optim.Adam([geometry_eval], lr=self.flags.lr)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop([geometry_eval], lr=self.flags.lr)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD([geometry_eval], lr=self.flags.lr)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_size=(128, 8))
        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Number of Parameters: {}".format(pytorch_total_params))
        return model

    def make_loss(self, logit=None, labels=None, G=None, return_long=False, epoch=None):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :param larger_BDY_penalty: For only filtering experiments, a larger BDY penalty is added
        :param return_long: The flag to return a long list of loss in stead of a single loss value,
                            This is for the forward filtering part to consider the loss
        :param pairwise: The addition of a pairwise loss in the loss term for the MD
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss
        BDY_loss = 0
        if G is not None:         # This is using the boundary loss
            X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
            X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
            relu = torch.nn.ReLU()
            BDY_loss_all = 1 * relu(torch.abs(G - self.build_tensor(X_mean)) - 0.5 * self.build_tensor(X_range))
            BDY_loss = 0.1*torch.sum(BDY_loss_all)

        self.MSE_loss = MSE_loss
        self.Boundary_loss = BDY_loss
        return torch.add(MSE_loss, BDY_loss)

    def build_tensor(self, nparray, requires_grad=False):
        return torch.tensor(nparray, requires_grad=requires_grad, device='cuda', dtype=torch.float)
    
    def get_train_loader(self):
        """
        Get the train loader from the self.data_x and self.data_y
        """
        if self.flags.dim_x == 1 and self.flags.dim_y == 1:
            DataSetClass = oned_oned
        elif self.flags.dim_x > 1 and self.flags.dim_y > 1:
            DataSetClass = nd_nd
        elif self.flags.dim_x == 1:
            DataSetClass = oned_nd
        else:
            DataSetClass = nd_oned
        train_data = DataSetClass(self.data_x, self.data_y)
        return torch.utils.data.DataLoader(train_data, batch_size=self.flags.batch_size)

    def make_optimizer(self, model_index, optimizer_type=None):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        # For eval mode to change to other optimizers
        if  optimizer_type is None:
            optimizer_type = self.flags.optim
        if optimizer_type == 'Adam':
            op = torch.optim.Adam(self.models[model_index].parameters(), lr=self.flags.lr,
                                weight_decay=self.flags.reg_scale)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop(self.models[model_index].parameters(), lr=self.flags.lr, 
                                weight_decay=self.flags.reg_scale)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD(self.models[model_index].parameters(), lr=self.flags.lr, 
                                weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def make_lr_scheduler(self, optm):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=self.flags.lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        for i in range(self.n_model):
            torch.save(self.models[i].state_dict(), os.path.join(self.ckpt_dir, 'best_model_{}.pt'.format(i)))

    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        for i in range(self.n_model):
            if torch.cuda.is_available():
                self.models[i].load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_{}.pt'.format(i))))
            else:
                self.models[i].load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_{}.pt'.format(i)),
                                         map_location=torch.device('cpu')))

    def train(self, model_ind):
        """
        The major training function. This would start the training using information given in the flags
        :param model_ind: The index of the model that would like to train
        :return: None
        """
        # Get the train loader from the data_x data_y
        train_loader = self.make_train_loader()

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.models[model_ind].cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer(model_ind)
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        a = self.flags.train_step
        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            # boundary_loss = 0                 # Unnecessary during training since we provide geometries
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU
                self.optm.zero_grad()                               # Zero the gradient first
                logit = self.model(geometry)                        # Get the output
                loss = self.make_loss(logit, spectra)               # Get the loss tensor
                loss.backward()                                     # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                train_loss += loss                                  # Aggregate the loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)

            if epoch % self.flags.eval_step == 0:                      # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                self.log.add_scalar('Loss/train', train_avg_loss, epoch)
                # self.log.add_scalar('Loss/BDY_train', boundary_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logit = self.model(geometry)
                    loss = self.make_loss(logit, spectra)                   # compute the loss
                    test_loss += loss                                       # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss.cpu().data.numpy() / (j+1)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss ))

                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model down...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)



class nd_nd(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind, :]

class nd_oned(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind]

class oned_nd(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind, :]

class oned_oned(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]
        