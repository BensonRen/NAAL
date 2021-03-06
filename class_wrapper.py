"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys
from numpy.core.defchararray import add
from numpy.lib.npyio import save

import pickle
from torch._C import dtype
# sys.path.append('../utils/')

from forward_models import simulator

# Torch
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from sklearn.metrics import pairwise_distances
from model_maker import NN, NAAL, Dropout_model

# Libs
import numpy as np
from math import inf
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def MSE(pred, truth, axis=1):
    """
    The MSE numpy version
    :param: pred/truth [#data, #dim]
    :return: list of MSE with len = #data
    """
    return np.mean(np.square(truth - pred), axis=axis)

class Network(object):
    def __init__(self, model_fn, flags, ckpt_dir=os.path.join(os.path.abspath(''), 'models')):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs    
        self.naal = flags.naal
        if flags.model_name is None:                    # leave custume name if possible
            self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        else:
            self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        # Make this directory
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.loss = self.make_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.n_model = flags.al_n_model                         # Save the number of models for quick reference
        self.dataset = flags.data_set                            # The dataset name 

        self.models = []
        # Creating the Committee
        if not self.naal:
            for i in range(flags.al_n_model):
                self.models.append(self.create_model())                        # The model itself
        else:
            self.models = [self.create_model()]
        
        # The dropout network
        if 'Dropout' in flags.al_mode:
            self.models.append(self.create_aux_model())
            print(self.models[-1])

        print('len of self.models = ', len(self.models))
        self.print_model_stats()                                # Print the stats

        # Setting up the simulator
        # self.simulator = self.init_simulator(self.flags.data_set)
        self.simulator = simulator

        # Creating the training dataset
        if flags.load_dataset is None:
            self.data_x, self.data_y = self.init_dataset(self.flags.al_n_x0)
        else:
            self.data_x, self.data_y =  self.load_dataset(flags.load_dataset, testset=False)
        
        # # Creating the random validation dataset
        # self.val_x, self.val_y = self.init_dataset(self.flags.al_test_n)        # Currently the validation set is the same size as test set

        # Creat the test set
        if flags.load_testset is None:
            self.test_X , self.test_Y = self.init_dataset(flags.al_test_n)
        else:
            self.test_X , self.test_Y = self.load_dataset(flags.load_testset)
        
        self.save_dataset(save_name=os.path.join('saved_datasets', self.flags.data_set), testset=True)   # Save the test set for reusing
        #print('dtype of train x', self.data_x.dtype)
        #print('dtype of train y', self.data_y.dtype)
        print('finish initializaiton')

        self.train_loss_tracker = [[] for i in range(self.n_model)]
        self.test_loss_tracker = [[] for i in range(self.n_model)]
        self.train_loss_tacker_epoch = [[] for i in range(self.n_model)]

        self.additional_X = None
        self.peak_spectra_meta()
        # quit()

    def create_aux_model(self):
        """
        The auxiliary model, now it is the dropout model
        """
        if 'Dropout' in self.flags.al_mode:         # THis is dropout mode
            dropout_model = Dropout_model(self.flags)
            print('creting dropout model')
            self.print_model_stats(dropout_model)
            return dropout_model

    def print_model_stats(self, model=None):
        """
        print the model statistics and total number of trainable paramter
        """
        if model is None:
            model = self.models[0]
        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Number of Parameters: {}".format(pytorch_total_params))

    # def init_simulator(self, data_set):
    #     """
    #     The function to initialize the simulator
    #     """
    #     if data_set == 'sine':
    #         freq = self.flags.freq
    #         def funct(x):
    #             return np.sin(x * freq)
    #     elif data_set == 'xsinx':
    #         freq = self.flags.freq
    #         def funct(x):
    #             return x * np.sin(x * freq)
    #     elif data_set == 'xsinsinx':
    #         freq = self.flags.freq
    #         def funct(x):
    #             return x * np.sin(3 * np.sin(x * freq))
    #     else:
    #         print("Your gt function is not defined!")
    #         exit()
    #     return funct

    def scale_uniform_to_data_distribution(self, X):
        """
        since the initialization of the backprop is uniform from [0,1], this function transforms that distribution
        to suitable prior distribution for each dataset. The numbers are accquired from statistics of min and max
        of the X prior given in the training set and data generation process
        :param X: The input uniform distribution from [0,1]
        :return: The transformed initial guess from prior distribution
        """
        X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
        X_new = X * X_range + X_lower_bound
        #print('after scaling dtype ', X_new.dtype)
        return X_new
    
    def get_boundary_lower_bound_uper_bound(self):
        """
        Due to the fact that the batched dataset is a random subset of the training set, mean and range would fluctuate.
        Therefore we pre-calculate the mean, lower boundary and upper boundary to avoid that fluctuation. Replace the
        mean and bound of your dataset here
        :return:
        """
        low = np.array(self.flags.dim_x_low, dtype=np.float32)
        high  = np.array(self.flags.dim_x_high, dtype=np.float32)
        return high-low, low, high

    def random_sample_X(self, number):
        """
        The function to randomly (uniformly) sample the X space according to the lower and upper bound
        """
        uniform_random = np.random.uniform(size=[number, self.flags.dim_x])
        uniform_random = uniform_random.astype(np.float32)
        return self.scale_uniform_to_data_distribution(uniform_random)

    def init_dataset(self, num):
        """
        The function to initiate and create the dataset
        """
        # Get the initialized x
        data_x = self.random_sample_X(num)
        data_y = self.simulator(self.dataset, data_x)
        return data_x, data_y

    def save_dataset(self, save_name=None, testset=False):
        """
        The function to save the dataset
        """
        # Make sure the dataset save to somewhere
        if save_name is None:
            save_name = os.path.join('dataset_saved', time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        if not os.path.isdir(save_name):    # Make sure the folder exists
            os.makedirs(save_name)
        if testset:
            np.save(os.path.join(save_name, 'test_x.npy'), self.test_X)
            np.save(os.path.join(save_name, 'test_y.npy'), self.test_Y)
        else:
            np.save(os.path.join(save_name, 'data_x.npy'), self.data_x)
            np.save(os.path.join(save_name, 'data_y.npy'), self.data_y)
        print("Your dataset has been saved to ", save_name)

    def load_dataset(self, load_dataset, testset=True):
        """
        The function to load the dataset
        """
        prefix = 'test' if testset else 'data'
        return np.load(os.path.join(load_dataset, self.flags.data_set, prefix + '_x.npy')),  np.load(os.path.join(load_dataset, self.flags.data_set, prefix + '_y.npy'))
    
    def save_flags(self, flags, save_dir, save_file="flags.obj"):
        """
        This function serialize the flag object and save it for further retrieval during inference time
        :param flags: The flags object to save
        :param save_file: The place to save the file
        :return: None
        """
        flags_dict = vars(flags)
        # Convert the dictionary into pandas data frame which is easier to handle with and write read
        with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
            print(flags_dict, file=f)
        with open(os.path.join(save_dir, save_file),'wb') as f:          # Open the file
            pickle.dump(flags, f)               # Use Pickle to serialize the object

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_size=(128, 8))
        return model

    def make_na_loss(self, logit, G, epoch=None, md=False):
        """
        The loss that the neural adjoint method use to do back propagation
        """
        # Get the VAR loss
        ensembled = torch.mean(logit, dim=0).unsqueeze(0).repeat(self.n_model, 1, 1)
        var = -1 * nn.functional.mse_loss(logit, ensembled)

        BDY_loss = 0
        if G is not None:         # This is using the boundary loss
            X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
            X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
            relu = torch.nn.ReLU()
            BDY_loss_all = 1 * relu(torch.abs(G - self.build_tensor(X_mean)) - 0.5 * self.build_tensor(X_range))
            BDY_loss = torch.sum(BDY_loss_all)
        if md:
            pairwise_dist_mat = torch.cdist(G, G, p=2)      # Calculate the pairwise distance
            MD_loss = torch.mean(torch.exp(-pairwise_dist_mat*50))
            # MD_loss = torch.mean(torch.pow((1/self.flags.na_md_radius)*relu(- pairwise_dist_mat + self.flags.na_md_radius)+1, 4)-1)       # The soft sphere exponential
            # MD_loss = torch.mean(relu(1/(pairwise_dist_mat + 1e-6) - 1/self.flags.na_md_radius))       # The soft sphere inverse potential model
            # MD_loss *= self.flags.na_md_coeff
            # Calculate the ratio of var loss and md loss
            with torch.no_grad():
                if MD_loss > 0:
                    var_md_loss_ratio = torch.abs(var/MD_loss)*50
                else:
                    var_md_loss_ratio = 1
            MD_loss *= var_md_loss_ratio
            # print('var loss = {}, MD_loss = {}, ratio = {} , BDY_loss = {}'.format(var, MD_loss, var_md_loss_ratio, BDY_loss))
            return var + BDY_loss + MD_loss, var, MD_loss/var_md_loss_ratio, BDY_loss # ALL LOSS
            # return BDY_loss + MD_loss, var, MD_loss/var_md_loss_ratio, BDY_loss  # Diversity only!!!
        self.Boundary_loss = BDY_loss
        return torch.add(var, BDY_loss), var, 0, BDY_loss

    def make_loss(self, logit=None, labels=None, G=None, return_long=False, epoch=None, total_batch_num=1):
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
        if self.naal:       # If NAAL, average the MSE to get the overall MSE for backprop
            if 'Drop' in self.flags.al_mode:        # This is for the dropout only
                # print('shape of logit and labels', logit.size(), labels.size())
                MSE_loss = nn.functional.mse_loss(logit, labels)
            else:
                # print('in make loss, logit size', logit.size())
                # print('in make loss, label size',labels.size())
                # print('naal')
                new_label = labels.unsqueeze(0).repeat(self.n_model, 1, 1)
                # print('shape of the new created label for naal:', new_label.size())
                MSE_loss = nn.functional.mse_loss(logit, new_label )
                # for i in range(self.flags.al_n_model):
                #     mse = nn.functional.mse_loss(logit[i, :, :], labels)
                #     MSE_loss += mse
                    #logit = torch.mean(logit, axis=0)
        else:
            # print('not naal')
            MSE_loss = nn.functional.mse_loss(logit, labels, reduction='mean')          # The MSE Loss
        self.MSE_loss = MSE_loss
        #print(MSE_loss.cpu().detach().numpy())
        return MSE_loss

    def build_tensor(self, nparray, requires_grad=False):
        return torch.tensor(nparray, requires_grad=requires_grad, device='cuda', dtype=torch.float)
    
    def get_loader(self, data_x, data_y):
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

        # shuffling process
        if self.flags.shuffle_each_model:
            shuffle_index = np.random.permutation(len(data_x))
            data_x = data_x[shuffle_index]
            data_y = data_y[shuffle_index]

        data = DataSetClass(data_x, data_y)
        # This is for solving the batch problem where the incomplete batch cause unstable training
        # if len(data_x) > self.flags.batch_size:
        #     return torch.utils.data.DataLoader(data, batch_size=self.flags.batch_size, shuffle=False, drop_last=True)
        # else:
        #     return torch.utils.data.DataLoader(data, batch_size=self.flags.batch_size, shuffle=False)#, drop_last=True)
        
        return torch.utils.data.DataLoader(data, batch_size=self.flags.batch_size, shuffle=False)#, drop_last=True)

    def make_optimizer(self, model_index, params=None, optimizer_type=None):
        """ finished
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        # For eval mode to change to other optimizers
        if  optimizer_type is None:
            optimizer_type = self.flags.optim
        # If the param is a parameter taking in
        if params is None:
            # THis is normal mode of training
            #print('params is none, model_index = {}'.format(model_index))
            params = self.models[model_index].parameters()
            # print(self.models[0])
            # print('model parameters')
            # print(self.models[0].parameters())
            # print('params', params)
            # print('make optimizer calling on index', model_index)
            # print('The total number of parameters to be optimized is ', sum(1 for _ in params))
            # print('The total number in the original list ', sum(1 for _ in self.models[0].parameters()))
            # print('plotting all the parameters')
            # for par in params:
            #     print(par)
            # print('finish printing params')
            # print('plotting the original one')
            # for par in self.models[0].parameters():
            #     print(par)
            # print("finish plotting the original model ")
            reg = self.flags.reg_scale
            lr = self.flags.lr
            # if 'Drop' in self.flags.al_mode:
            #     lr = 0.001
        else:
            # THis is NAAL part of the finding large variance
            lr = self.flags.nalr
            reg = 0
        

        if optimizer_type == 'Adam':
                op = torch.optim.Adam(params, lr=lr, weight_decay=reg)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop(params, lr=lr, weight_decay=reg)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD(params, lr=lr, weight_decay=reg)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op
    
    def make_NA_lr_scheduler(self, optm, verbose=False):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=self.flags.na_lr_decay_rate, patience=5, verbose=verbose, threshold=1e-6)

    def make_lr_scheduler(self, optm, verbose=False):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=self.flags.lr_decay_rate,
                                              patience=10, verbose=verbose, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        for i in range(self.n_model):
            self.save_single(i)
    
    def save_single(self, i):
        """
        Save single model
        """
        torch.save(self.models[i].state_dict(), os.path.join(self.ckpt_dir, 'best_model_{}.pt'.format(i)))
    
    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        for i in range(self.n_model):
            self.load_single(i)

    def load_single(self, i):
        if torch.cuda.is_available():
            self.models[i].load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_{}.pt'.format(i))))
        else:
            self.models[i].load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_{}.pt'.format(i)),
                                        map_location=torch.device('cpu')))

    def train_single(self, model_ind, verbose=True):
        """ (finished naal)
        The major training function. This would start the training using information given in the flags
        :param model_ind: The index of the model that would like to train
        :return: None
        """
        best_validation_loss = inf

        # Get the train loader from the data_x data_y
        if self.flags.bootstrap > 0:
            print('bootstrping!')
            random_permutation = np.random.permutation(len(self.data_x))
            index_end = int(self.flags.bootstrap * len(self.data_x))
            train_loader = self.get_loader(self.data_x[random_permutation[:index_end]], self.data_y[random_permutation[:index_end]])
        else:
            train_loader = self.get_loader(self.data_x, self.data_y)
        
        # Debugging, using the train set as the validation loader
        val_loader = self.get_loader(self.test_X, self.test_Y)
        # val_loader = self.get_loader(self.data_x, self.data_y)

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.models[model_ind].cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer(model_ind)
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        total_batch_num = len(train_loader)
        epoch = 0
        while epoch < self.flags.train_step:
        # for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            self.models[model_ind].train()
            self.optm.zero_grad()
            for j, (X, Y) in enumerate(train_loader):
                # print('Epoch {} batch {}, size X {}, size Y {}'.format(epoch, j, X.size(), Y.size()))
                if cuda:
                    X = X.cuda()                                    # Put data onto GPU
                    Y = Y.cuda()                                    # Put data onto GPU
                # if epoch == 0 and model_ind == 0:
                #     print('batch = {}, size = {}'.format(j, len(X)))
                # if X.size(0) < 0.5*self.flags.batch_size:
                #     for name ,child in (self.models[model_ind].named_children()):
                #         if name.find('BatchNorm') != -1:
                #             for param in child.parameters():
                #                 param.requires_grad = False
                #         else:
                #             for param in child.parameters():
                #                 param.requires_grad = True 
                #     if epoch == 0 and model_ind == 0:
                #         print('switching to eval model for imcomplete batch of size {}'.format(X.size(0)))
                #self.optm.zero_grad()                               # Zero the gradient first
                logit = self.models[model_ind](X.float())                    # Get the output
                loss = self.make_loss(logit, Y.float())# total_batch_num=total_batch_num)                     # Get the loss tensor
                #loss /= total_batch_num
                # print('Epoch {} batch {}, loss = {}'.format(epoch, j, loss.cpu().detach().numpy()))
                loss.backward()                                     # Calculate the backward gradients
                train_loss += loss.cpu().data.numpy()                                  # Aggregate the loss
            
                # Batch effect debugging
                self.optm.step()                                     # Move one step the optimizer
                self.optm.zero_grad()
                epoch += 1

            # Calculate the avg loss of training
            train_avg_loss = train_loss / total_batch_num
            
            ##################################
            # Debug 11.06 of the batch issue #
            ##################################
            # self.train_loss_tracker[model_ind].append(train_avg_loss)
            # self.train_loss_tacker_epoch[model_ind].append(total_batch_num)

            if epoch % self.flags.eval_step == 0:                   # For eval steps, do the evaluations and tensor board
                # Set to Evaluation Mode
                self.models[model_ind].eval()
                #print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (X, Y) in enumerate(val_loader):       # Loop through the eval set
                    if cuda:
                        X = X.cuda()
                        Y = Y.cuda()
                    logit = self.models[model_ind](X.float())
                    loss = self.make_loss(logit, Y)                 # compute the loss
                    test_loss += loss                               # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss.cpu().data.numpy() / (j+1)
                
                # track the test loss
                # self.test_loss_tracker[model_ind].append(test_avg_loss)
                if verbose:
                    print("For model %d, this is Epoch %d, training loss %.5f, validation loss %.5f" \
                        % (model_ind, epoch, train_avg_loss, test_avg_loss ))

                # Model improving, save the model down
                if test_avg_loss < best_validation_loss:
                    best_validation_loss = test_avg_loss
                    #self.save_single(model_ind)
                    #print("Saving the model down...")
            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        
        # After the training, take the best performing one from loading it
        #self.load_single(model_ind)

    def train(self):
        """ (finished naal)
        Aggregate function of the training all models
        """
        if not self.naal:
            for i in range(self.n_model):
                # print('training model ', i)
                self.train_single(i)
        else:
            # print('training model 0')
            self.train_single(0)
            
            # Train the auxilary network of dropout as well
            if 'Dropout' in self.flags.al_mode:
                self.train_single(1)

    def eval_model(self, model_ind, eval_X, eval_Y):
        """
        Evaluation of 
        """
        print('doing evaluationg on model ', model_ind )
        Ypred = self.pred_model(model_ind, eval_X, output_numpy=True)
        mse = MSE(Ypred, eval_Y)
        print('model {} has mse = {}'.format(model_ind, np.mean(mse)))
        return mse
    
    def pred_dropout_model(self, model_ind, test_X, output_numpy=False, dropout=False):
        """ (finished naal)
        Output the prediction of model[model_ind]
        """
        # Get a tensor version of the test X instead of a numpy 
        if isinstance(test_X, np.ndarray):
            test_X = self.build_tensor(test_X)
        
        # Move things to the GPU
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            test_X = test_X.cuda()
            self.models[model_ind].cuda()

        # Start predicting, set everything into evaluation mode
        self.models[model_ind].eval()
        # For all the dropout layers, we are still doing dropouts during evaluation stage
        if dropout:
            for m in self.models[model_ind].modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        # Do the prediction function
        # print('size of test_X is', test_X.size())
        Ypred = self.dropout_model(test_X.float())

        # Convert to numpy if necessary
        if output_numpy:
            Ypred = Ypred.cpu().data.numpy()
        return Ypred

    def pred_model(self, model_ind, test_X, output_numpy=False, dropout=False):
        """ (finished naal)
        Output the prediction of model[model_ind]
        """
        # Get a tensor version of the test X instead of a numpy 
        if isinstance(test_X, np.ndarray):
            test_X = self.build_tensor(test_X)
        
        # Move things to the GPU
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            test_X = test_X.cuda()
            self.models[model_ind].cuda()

        # Start predicting, set everything into evaluation mode
        self.models[model_ind].eval()
        # For all the dropout layers, we are still doing dropouts during evaluation stage
        if dropout:
            for m in self.models[model_ind].modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        # Do the prediction function
        # print('size of test_X is', test_X.size())
        Ypred = self.models[model_ind](test_X.float())

        # Convert to numpy if necessary
        if output_numpy:
            Ypred = Ypred.cpu().data.numpy()
        return Ypred

    def ensemble_predict_mat(self, test_X, num_repeat=25, VAR=False):
        """ (finished naal)
        Get each model to predict and output a large matrix
        :param num_repate: Number of repetition for dropout method
        :param VAR: This ensemble calculation is for Variance calculation (different for dropout)
        """
        if 'Drop' in self.flags.al_mode and VAR:
            # Initialize the Ypred_mat
            Ypred_mat = np.zeros([num_repeat, len(test_X), self.flags.dim_y])
            # Repeat the prediction for 
            for i in range(num_repeat):
                Ypred_mat[i, :, :] = self.pred_model(1, test_X, output_numpy=True,dropout=True)
            return Ypred_mat

        if self.naal:
            # print('doing ensemble predict for naal')
            return self.pred_model(0, test_X, output_numpy=True)

        Ypred_mat = np.zeros([self.n_model, len(test_X), self.flags.dim_y])
        for i in range(self.n_model):
            Yp = self.pred_model(i, test_X, output_numpy=True)
            Ypred_mat[i, :, :] = Yp
        return Ypred_mat

    def ensemble_predict(self, test_X):
        """
        Get the average output Y for a given dataset test_X
        """
        # if self.flags.al_mode == 'Dropout':
        #     return self.pred_model(0, test_X, output_numpy=True, dropout=False)
        # else:
        Ypred_mat = self.ensemble_predict_mat(test_X, VAR=False)       # Get the Ypred mat from each model prediction
        return np.mean(Ypred_mat, axis=0)
    
    def ensemble_MSE(self, test_X, test_Y):
        """
        Get the MSE for the ensemble model (data point wise)
        """
        Ypred = self.ensemble_predict(test_X)     # Get the Ypred mat from each model prediction
        return MSE(Ypred, test_Y)
    
    def ensemble_VAR(self, test_X):
        """
        Get the Variance for the ensemble model
        param: num_repeat (default = 25) Number of times to do the prediction to estimate the variance
        """
        Ypred_mat = self.ensemble_predict_mat(test_X, VAR=True)
        mean_pred = np.mean(Ypred_mat, axis=0)
        var = np.mean(np.mean(np.square(Ypred_mat - mean_pred),axis=0), axis=-1)
        # print('the shape of the variance output is ', np.shape(var))
        return var, Ypred_mat

    def add_X_into_trainset(self, additional_X, additional_Y=None, plot_first_y=False, save_dir=None, step_num=None):
        """
        Add the additional_X (optinal additional_Y) into the training set to self.data_x and self.data_y
        They are all in numpy format
        plot_first_y: Plotting the first Y value to look at it, debugging purpose
        """
        # Save those extra data_x if
        if save_dir:
            np.save(os.path.join(save_dir, 'data_x_step_{}.npy'.format(step_num)), additional_X)

        # Simulate Y if it is not provided
        if additional_Y is None:
            additional_Y = self.simulator(self.dataset, additional_X)
        # print('during adding X into train set, the shape of X and Y are', np.shape(additional_X), np.shape(additional_Y))
        if plot_first_y:        # Debugging
            plt.plot(additional_Y[0, :])
            plt.savefig('sample_Y.png')
        self.data_x = np.concatenate([self.data_x, additional_X])
        self.data_y = np.concatenate([self.data_y, additional_Y])

    def get_additional_X(self, save_dir=None, step_num=None):
        """
        Select the additional X from a pool (that is randomly generated)
        """
        def NormalizeData(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        pool_x = self.random_sample_X(self.flags.al_x_pool)                 # Generate some random samples for making the pool
        #if step_num != None:
        #    print('in step {}, the sum of pool x is {}'.format(step_num, np.sum(pool_x)))
        pool_x_pred_y, pool_mse_mean, pool_chosen_one_mse, var_mse_coreff, tau = 0, 0, 0, 0, 0     # in case it is not MSE based
        if 'NA' not in self.flags.al_mode and 'Core' not in self.flags.al_mode:
            pool_y = self.simulator(self.dataset, pool_x)
            pool_x_pred_y = self.ensemble_predict(pool_x)    # make ensemble predictions
        if self.flags.al_mode == 'MSE':
            pool_mse = MSE(pool_x_pred_y, pool_y)                               # rank the ensembled prediction and get the top ones 
            index = np.argsort(pool_mse)
            # prove that we are actually choosing the most outstanding ones
            pool_mse_mean = np.mean(pool_mse)
            pool_chosen_one_mse = np.mean(pool_mse[index[-self.flags.al_n_dx:]])
            #print('the mean mse of the whole pool is {}'.format(pool_mse_mean))
            #print('the mean mse of chosen ones {}'.format(pool_chosen_one_mse))
        elif  'VAR' in self.flags.al_mode or 'Dropout' in self.flags.al_mode:
            pool_VAR, pool_x_pred_y_mat = self.ensemble_VAR(pool_x)
            index = np.argsort(pool_VAR)
            if self.flags.plot_correlation_VAR_MSE:
                # print('shape of mat', np.shape(pool_x_pred_y_mat))
                pool_mse_models = np.mean(MSE(pool_x_pred_y_mat, pool_y, axis=0), axis=-1)                              # rank the ensembled prediction and get the top ones 
                # print('size of pool_mse_models', np.shape(pool_mse_models))
                # print('size of pool_VAR', np.shape(pool_VAR))
                # print(pool_mse_models)
                f = plt.figure(figsize=[8, 4])
                # Calculate the R coeff
                var_mse_coreff = np.corrcoef(pool_mse_models, pool_VAR)[0, 1]
                # Calculate the Tau coeff
                tau, p_value = stats.kendalltau(pool_mse_models, pool_VAR)
                plt.scatter(pool_mse_models, pool_VAR,label='R={:.2f} Tau={:.2f}'.format(var_mse_coreff, tau))
                plt.xlabel('pool mse')
                plt.ylabel('pool VAR')
                plt.title('VAR_MSE correlation @ step {}'.format(step_num))
                plt.legend()
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(os.path.join(save_dir, 'VAR_MSE_correlation_step{}.png'.format(step_num)))
            
            if 'Core' in self.flags.al_mode:
                """
                QBC sift till there are 2 dx number of points left, and core-set always choose dx from 2dx proposals
                """
                X_train, x_pool, X_add = self.data_x, pool_x[index[-self.flags.al_n_dx*2:]], np.zeros([self.flags.al_n_dx, self.flags.dim_x])
                for i in range(self.flags.al_n_dx):        # Adding the points one-by-one
                    dist = pairwise_distances(X_train, x_pool, metric='euclidean')      # Get the pair-wise distance
                    # print('shape of dist mat', np.shape(dist))
                    D_min = np.min(dist, axis=0).reshape(-1, )                         # Take the min (closest neighbor)
                    # print('shape of D_min', np.shape(D_min))
                    max_min_D = np.argmax(D_min)                                        # Get the one with the furthest neighbour
                    X_add[i, :] = x_pool[max_min_D, :]                                  # Add this to the list
                    #X_add.append(x_pool[np.argmax(D_min), :])                          # Add this to the list
                    X_train = np.vstack((X_train, x_pool[max_min_D, :]))                # Take this into the training set
                    # print('shape of X_train', np.shape(X_train))
                    x_pool = np.delete(x_pool, np.argmax(D_min), 0)                     # Remove this added one 
                pool_x = X_add      # Update the pool
                pool_y = self.simulator(self.dataset, pool_x)
                index = range(len(pool_x))
            elif 'Div_Den' in self.flags.al_mode:
                """
                QBC with diversity and density in "Qeury-by-committee improvement with diversity and density in batch active learning", 2018
                Diversity only would be taking w_den = 0
                Density only would be taking w_div = 0
                """
                X_train, x_pool, X_add = self.data_x, np.copy(pool_x), np.zeros([self.flags.al_n_dx, self.flags.dim_x])
                # Normalize pool_VAR
                pool_VAR = NormalizeData(pool_VAR)
                for i in range(self.flags.al_n_dx):        # Adding the points one-by-one
                    #############
                    # Diversity #
                    #############
                    if self.flags.w_div > 0:
                        dist = pairwise_distances(X_train, x_pool, metric='euclidean')      # Get the pair-wise distance
                        # print('shape of dist mat', np.shape(dist))
                        D_min = np.min(dist, axis=0).reshape(-1, )                         # Take the min (closest neighbor)
                        # Normalize D_min into [0, 1] by dividing the largest
                        D_min = NormalizeData(D_min)
                        ## Work the cosine similarity now
                    else:
                        D_min = np.zeros_like(pool_VAR)
                    #############
                    # Density   #
                    #############
                    if self.flags.w_den > 0:
                        # First get the pairwise distance
                        dist = pairwise_distances(x_pool, x_pool, metric='cosine')
                        # Sort he distance matrix
                        dist.sort(axis=1)
                        # Calculate the similarity score by taking averageg of the 10 NN (judged by cosine)
                        similarity = np.mean(dist[:, :10], axis=1)
                        # Normalize similarity
                        similarity = NormalizeData(similarity)
                    else:
                        similarity = np.zeros_like(pool_VAR)

                    # print('shape of D_min', np.shape(D_min))
                    # print('shape of pool_VAR', np.shape(pool_VAR))
                    Div_n_Var_metric = (1-self.flags.w_div - self.flags.w_den) * pool_VAR + self.flags.w_div * D_min + self.flags.w_den * similarity
                    # print('shape of Div_n_Var_metric', np.shape(Div_n_Var_metric))
                    max_min_D = np.argmax(Div_n_Var_metric)                             # Get the one with the furthest neighbour
                    # print('shape of x_pool {}'.format(np.shape(x_pool)))
                    X_add[i, :] = x_pool[max_min_D, :]                                  # Add this to the list
                    #X_add.append(x_pool[np.argmax(D_min), :])                          # Add this to the list
                    X_train = np.vstack((X_train, x_pool[max_min_D, :]))                # Take this into the training set
                    # print('shape of X_train', np.shape(X_train))
                    x_pool = np.delete(x_pool, max_min_D, 0)          # Remove this added one 
                    pool_VAR = np.delete(pool_VAR, max_min_D, 0)      # Remove this added one in the pool as well
                pool_x = X_add      # Update the pool
                pool_y = self.simulator(self.dataset, pool_x)
                index = range(len(pool_x))

        elif self.flags.al_mode == 'Random':
            # Two ways of random, the first is to permute as below, however, this would interupt the random state of numpy, therefore for reproducibility we use the other
            # index = np.random.permutation(len(pool_x))
            # The simplest random way, just the sequence itself
            index = range(len(pool_x))
        elif self.flags.al_mode == 'Core-set':
                # # Setting up the pool, labelled set and the place holder for the chosen ones 
                X_train, x_pool, X_add = self.data_x, pool_x, np.zeros([self.flags.al_n_dx, self.flags.dim_x])
                for i in range(self.flags.al_n_dx):        # Adding the points one-by-one
                    dist = pairwise_distances(X_train, x_pool, metric='euclidean')      # Get the pair-wise distance
                    # print('shape of dist mat', np.shape(dist))
                    D_min = np.min(dist, axis=0).reshape(-1, 1)                         # Take the min (closest neighbor)
                    # print('shape of D_min', np.shape(D_min))
                    max_min_D = np.argmax(D_min)                                        # Get the one with the furthest neighbour
                    X_add[i, :] = x_pool[max_min_D, :]                                  # Add this to the list
                    #X_add.append(x_pool[np.argmax(D_min), :])                          # Add this to the list
                    X_train = np.vstack((X_train, x_pool[max_min_D, :]))                # Take this into the training set
                    # print('shape of X_train', np.shape(X_train))
                    x_pool = np.delete(x_pool, np.argmax(D_min), 0)                     # Remove this added one 
                pool_x = X_add      # Update the pool
                pool_y = self.simulator(self.dataset, pool_x)
                index = range(len(pool_x))
        elif 'NA' in self.flags.al_mode:
            """
            The case where the Neural adjoint method is used for getting the additional output
            """
            # Initialize the raw pool
            na_pool_raw = torch.rand([self.flags.na_num_init, self.flags.dim_x], requires_grad=True, device='cuda')
            # Get the pool to have training distribution
            X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
            # Get the optimizer
            self.optm_na = self.make_optimizer(model_index=0, params=[na_pool_raw])
            self.lr_scheduler_na = self.make_NA_lr_scheduler(self.optm_na)
            # Set to evaluation mode
            self.models[0].eval()
            # Record the starting point
            na_start_pool = na_pool_raw.cpu().detach().numpy() * 2 - 1
            
            # MD switch
            if 'MD' in self.flags.al_mode:
                md = True
            else:
                md = False
            
            # Record the different losses
            var_loss_list, md_loss_list, bdy_loss_list = [], [], []
            var_max = 0         # Initialize the starting var loss (var goes down from 0)
            # Start the back propagating process
            for i in range(self.flags.naal_steps):
                na_pool = na_pool_raw * self.build_tensor(X_range) + self.build_tensor(X_lower_bound)
                # Calculate the VAR and see it going up
                # pool_VAR, pool_x_pred_y_mat = self.ensemble_VAR(na_pool.cpu().detach().numpy())
                # print('in NAAL, epoch {} VAR = {}'.format(i, np.mean(pool_VAR)))
                self.optm_na.zero_grad()
                logit = self.models[0](na_pool)
                loss, var_loss, md_loss, bdy_loss = self.make_na_loss(logit, G=na_pool, md=md)
                # print('retaining graph')
                loss.backward(retain_graph=True)
                self.optm_na.step()

                if var_loss < var_max:  # This is lower, add this 
                    pool_x = na_pool.cpu().detach().numpy()
                    var_max = var_loss
                # Record the best one
                # print('in AL step {}, NA step {}, VAR = {}'.format(step_num, i, var_loss.detach().cpu().numpy()))
                # Debugging purpose code:
                # print('debugging in NAMD backproping')
                if md:
                    var_loss_list.append(var_loss.detach().cpu().numpy())
                    md_loss_list.append(md_loss.detach().cpu().numpy())
                    bdy_loss_list.append(bdy_loss.detach().cpu().numpy())
                # print(logit)
            
            # Calculate the average distance travelled (excluding the things wandered out)
            na_final = na_pool.cpu().detach().numpy()
            dist = na_start_pool[np.abs(na_final) < 1] - na_final[np.abs(na_final) < 1] 
            avg_dist = np.mean(np.abs(dist))
            with open(os.path.join(save_dir, 'NA_dist_traveled.txt'), 'a') as f:
                f.write('AL step: {}, average distance NA traveled: {} \n'.format(step_num, avg_dist))
            if md: # Some diagonostics for NAMD method for plotting
                ####
                # !!! after taking the max var point, this chunk of code can be wrong #
                # Plot the loss
                f = plt.figure()
                ax = plt.subplot(211)
                plt.plot(var_loss_list,label='var')
                plt.legend()
                ax = plt.subplot(212)
                plt.plot(md_loss_list,label='md')
                plt.legend()
                plt.savefig(os.path.join(save_dir, 'NAMD backprop at step {} .png'.format(step_num)))

            if 'Core' in self.flags.al_mode:
                ####
                # !!! after taking the max var point, this chunk of code can be wrong #
                X_train, x_pool, X_add = self.data_x, na_pool.cpu().detach().numpy(), np.zeros([self.flags.al_n_dx, self.flags.dim_x])
                for i in range(self.flags.al_n_dx):        # Adding the points one-by-one
                    dist = pairwise_distances(X_train, x_pool, metric='euclidean')      # Get the pair-wise distance
                    # print('shape of dist mat', np.shape(dist))
                    D_min = np.min(dist, axis=0).reshape(-1, 1)                         # Take the min (closest neighbor)
                    # print('shape of D_min', np.shape(D_min))
                    max_min_D = np.argmax(D_min)                                        # Get the one with the furthest neighbour
                    X_add[i, :] = x_pool[max_min_D, :]                                  # Add this to the list
                    #X_add.append(x_pool[np.argmax(D_min), :])                          # Add this to the list
                    X_train = np.vstack((X_train, x_pool[max_min_D, :]))                # Take this into the training set
                    # print('shape of X_train', np.shape(X_train))
                    x_pool = np.delete(x_pool, np.argmax(D_min), 0)                     # Remove this added one 
                pool_x = X_add      # Update the pool
                pool_y = self.simulator(self.dataset, pool_x)
                index = range(len(pool_x))
            else:    
                # Finished the backprop, get the list
                # pool_x = na_pool.cpu().detach().numpy()
                # This is to make sure all pool_x are within the range, although we have the boundary loss, currently there are still very small of them
                pool_x[pool_x > 1] = 1 - 0.005* np.random.random(1)              # Effectively make it close to the boundary
                pool_x[pool_x < -1] = -1 + 0.005* np.random.random(1)            # Effectively make it close to the boundary
                #pool_x_random_within = np.random.random(size=np.shape(pool_x)) * 2 - 1
                #pool_x[np.abs(pool_x) > 1] = pool_x_random_within[np.abs(pool_x) > 1]
                assert np.sum(np.abs(pool_x) > 1) == 0, 'There are points out of domain, check!'
                pool_y = self.simulator(self.dataset, pool_x)
                ensembled = torch.mean(logit, dim=0).unsqueeze(0).repeat(self.n_model, 1, 1)
                var = nn.functional.mse_loss(logit, ensembled, reduction='none').cpu().detach().numpy()
                # print('var size', np.shape(var))
                var = np.reshape(np.mean(np.mean(var, axis=0), axis=-1), [-1, ])
                # print('after var size', np.shape(var))
                # print('shape of pool', np.shape(pool_x))
                index = np.argsort(var)        # Choosing the best k ones
                # Plotting the vairance distribution of the points selected in the current active learning step
                if False:
                    import seaborn as sns
                    f = plt.figure(figsize=[8, 4])
                    plt.hist(var)
                    # Save the figure
                    plt.savefig(os.path.join(save_dir, 'variance_distribution_step_{}.png'.format(step_num)))
                    f = plt.figure(figsize=[8, 8])
                    for dim_num in range(self.flags.dim_x):
                        # ax = plt.subplot('{}1{}'.format(self.dim_x+1, dim_num+1))
                        sns.distplot(pool_x[:, dim_num], hist=False, color='C{}'.format(dim_num),
                                    kde=True)
                        sns.distplot(pool_x[index[-self.flags.al_n_dx:], dim_num], hist=False, 
                                    color='C{}'.format(dim_num), kde=True,kde_kws={'linestyle': '--'})
                    # plt.xlim([-1, 1])
                    plt.savefig(os.path.join(save_dir, 'dimension_wise_distribution_step_{}.png'.format(step_num)))
                    plt.close('all')
                # index = range(len(pool_x))
        else:
            print('Your Active Learning mode is wrong, check again!')
            1/0     # Manuall stop to print the stack as well
        return pool_x[index[-self.flags.al_n_dx:]], pool_y[index[-self.flags.al_n_dx:]], pool_x_pred_y, pool_y, index, pool_mse_mean, pool_chosen_one_mse, var_mse_coreff, tau

    def add_noise_initialize(self, noise_factor=2):
        """
        This function magnifies the noise for the initialized network by directly multiplying all the weights, since it is a non-linear system it is more noisy
        """
        for i in range(self.n_model):
            for module_list in self.models[i].children():
                for layer in module_list:
                    with torch.no_grad():
                        try:
                            layer.weight *= noise_factor
                            layer.bias *= noise_factor
                            # layer.weight += np.random.normal(0, noise_factor)
                            # layer.bias += np.random.normal(0, noise_factor)
                        except:
                            print('In add noise init, this is layer {}, this is not working'.format(layer))
        
    def active_learn(self, trail=0):
        """
        The main active learning function
        """
        test_set_mse, train_set_mse, mse_pool, mse_selected_pool, mse_selected_after_train, var_mse_coreff_list, var_mse_tau = [], [], [], [], [], [], []
        # Active learning part
        al_step, num_good_to_stop = 0, 0    # Initialize some looping variables
        try: 
            al_mode_str = self.flags.al_mode
            if 'NA' in al_mode_str:
                al_mode_str += 'init_{}_nalr_{}_decay_{}'.format(self.flags.na_num_init, self.flags.nalr, self.flags.na_lr_decay_rate)
            elif 'Drop' in al_mode_str:
                al_mode_str += 'p_{}'.format(self.flags.dropout_p)
            save_dir = os.path.join(self.flags.plot_dir,
            '{}_{}_retrain_{}_complexity_{}_bs_{}_pool_{}_dx_{}_step_{}_x0_{}_nmod_{}_trail_{}'.format(self.flags.data_set,
            al_mode_str, self.flags.reset_weight, len(self.flags.linear) - 2, self.flags.batch_size,self.flags.al_x_pool, 
            self.flags.al_n_dx, self.flags.al_n_step, self.flags.al_n_x0, self.flags.al_n_model, trail))
        except:
            save_dir = 'results/fig/{}_{}_retrain_{}_complexity_{}_bs_{}_pool_{}_dx_{}_step_{}_x0_{}_nmod_{}_trail_{}'.format(self.flags.data_set, 
            self.flags.al_mode, self.flags.reset_weight, len(self.flags.linear) - 2, self.flags.batch_size,self.flags.al_x_pool, self.flags.al_n_dx, 
            self.flags.al_n_step, self.flags.al_n_x0, self.flags.al_n_model, trail)
        # Make sure this is not missed
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        # save the starting data x
        np.save(os.path.join(save_dir, 'data_x_start.npy'), self.data_x)

        while num_good_to_stop < self.flags.stop_criteria_num and al_step < self.flags.al_n_step_cap:
        #for al_step in range(self.flags.al_n_step):

            # Adding the noise in the network prior
            # if al_step == 0:
            #     self.add_noise_initialize()

            # reset weights for training
            if self.flags.reset_weight:
                self.reset_params()

            # Train again here
            self.train()

            if self.flags.plot:# and 'robo' not in self.flags.data_set:
                self.plot_both_plots(iteration_ind=al_step, save_dir=save_dir)                     # Get the trained model

            # Select the subset that is the best behaving
            if al_step > 0:
                mse_added = np.mean(self.ensemble_MSE(additional_X, self.simulator(self.dataset, additional_X)))
                #print('after training, the MSE of the added X is ', mse_added)
                mse_selected_after_train.append(mse_added)
            
            # Calculate mse and report that
            mse_train = np.mean(self.ensemble_MSE(self.data_x, self.data_y))
            mse_test = np.mean(self.ensemble_MSE(self.test_X, self.test_Y))
            print('AL step {}, current train set size = {}, train set mse = {:3e}, test set mse = {:3e}, the AL_mode is {}, retrain = {}'.format(al_step, 
                    len(self.data_x), mse_train, mse_test, self.flags.al_mode, self.flags.reset_weight))
            
            # First we select the additional X
            additional_X, additional_Y, pool_x_pred_y, pool_y, index, pool_mse, pool_chosen_mse, var_mse_coreff, tau = self.get_additional_X(save_dir=save_dir, step_num=al_step)
            
            # Put them into training set
            self.add_X_into_trainset(additional_X, additional_Y, save_dir=save_dir, step_num=al_step)
            self.additional_X = additional_X 
            
            # Adding things to list for post processing
            test_set_mse.append(mse_test)
            train_set_mse.append(mse_train)
            mse_pool.append(pool_mse)
            mse_selected_pool.append(pool_chosen_mse)
            var_mse_coreff_list.append(var_mse_coreff)
            var_mse_tau.append(tau)
            plt.close('all')

            # Stopping control
            al_step += 1
            
            # Compare with stopping criteria, if it is lower than the stopping criteria, shout out and add the stop count
            if mse_test < self.flags.mse_cutoff:
                num_good_to_stop += 1
                print('Bingo! @AL step {} your network test error is {}, better than stopping criteria {},\
                     this is the {}-th time'.format(al_step, mse_test, self.flags.mse_cutoff, num_good_to_stop))
            
           
        # Plot the post analysis plots
        self.plot_analysis_mses(test_set_mse, train_set_mse, mse_pool, mse_selected_pool, 
                                mse_selected_after_train, var_mse_coreff_list, var_mse_tau, save_dir=save_dir)
        
        # Plot the final Xtrain distribution
        self.get_training_data_distribution(iteration_ind='end', save_dir=save_dir)
        
        # self.plot_train_loss_tracker(save_dir=save_dir)

        # Save the flag for record purpose
        self.save_flags(self.flags, save_dir)

        # Make sure all the figures are closed
        plt.close('all')
        
    def plot_analysis_mses(self, test_set_mse, train_set_mse, mse_pool, mse_selected_pool, 
                        mse_selected_after_train, var_mse_coreff_list, var_mse_tau, save_dir, save_raw_data=True):
        """
        The plotting function for the post analysis
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if self.flags.plot:
            f = plt.figure()
            plt.plot(test_set_mse, '--x', alpha=0.3, linewidth=4,  label='test set')
            plt.plot(train_set_mse, alpha=0.4, label='train set')
            plt.plot(mse_selected_after_train, '--x', alpha=0.5, linewidth=2, label='selected after train')
            
            if self.flags.al_mode == 'MSE':
                plt.plot(mse_pool, alpha=0.5, label='pool mse')
                plt.plot(mse_selected_pool, alpha=0.5, label='selected in pool')
            plt.legend()
            plt.xlabel('iteration')
            plt.ylabel('MSE')
            plt.title('MSE comparison')
            plt.savefig(os.path.join(save_dir, 'agg_mse_plot.png'))

            # Plot the correlation points
            if np.sum(var_mse_coreff_list) != 0:
                f = plt.figure(figsize=[8, 4])
                plt.plot(var_mse_coreff_list,label='R')
                plt.plot(var_mse_tau, label='tau')
                plt.xlabel('iteration')
                plt.ylabel('mse-var cor')
                plt.legend()
                plt.savefig(os.path.join(save_dir, 'mse_var_cor.png'))

        if save_raw_data:           # The option to save the raw data
            np.save(os.path.join(save_dir, 'test_mse'), test_set_mse)
            np.save(os.path.join(save_dir,'train_mse'), train_set_mse)
            np.save(os.path.join(save_dir,'mse_selected_after_train'), mse_selected_after_train)
            if self.flags.al_mode == 'MSE':
                np.save(os.path.join(save_dir,'mse_pool'), mse_pool)
                np.save(os.path.join(save_dir, 'mse_selected_pool'), mse_selected_pool)
            elif self.flags.al_mode == 'VAR':
                np.save(os.path.join(save_dir,'var_mse_coreff'), var_mse_coreff_list)
                np.save(os.path.join(save_dir,'var_mse_tau'), var_mse_tau)

    def reset_params(self):
        """
        The funciton to reset all the trainable parameters
        """
        def weights_init(m):
            if isinstance(m, nn.Linear):
                # print('resetting weight')
                torch.nn.init.xavier_uniform_(m.weight.data)

        for i in range(self.n_model):
            if self.naal and i > 0:
                continue
            self.models[i].apply(weights_init)
            # for layer in self.models[i].children():
            #     if hasattr(layer, 'reset_parameters'):
            #         print('resetting weight')
            #         layer.reset_parameters()

    #########################################################
    # The portion that is used to debug the training process#
    #########################################################
    def get_training_data_distribution(self, iteration_ind, save_dir, fig_ax=None):
        """
        The function that output the plot of the histogram plot of the training data distribution
        """
        if fig_ax is None:
            f = plt.figure(figsize=[10, 3])
        else:
            ax = plt.subplot(211)
        plt.hist(self.data_x, bins=100)
        if self.additional_X is not None:
            plt.hist(self.additional_X, bins=100, label='added', alpha=0.4)
        # plt.xlim([self.flags.dim_x_low[0], self.flags.dim_x_high[0]])
        plt.xlabel('x')
        plt.ylabel('frequency')
        plt.title('training data distribution @ iteration {}, total #= {} '.format(iteration_ind, len(self.data_x)))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if fig_ax is None:
            plt.savefig(os.path.join(save_dir, 'train_distribution_@iter_{}'.format(iteration_ind)))
        # Save the data_x distribution into numpy file
        np.save(os.path.join(save_dir, 'final_x.npy'), self.data_x)

    def plot_sine_debug_plot(self, iteration_ind, save_dir, fig_ax=None):
        """
        The function that plots the 
        """
        if fig_ax is None:
            f = plt.figure(figsize=[10, 3])
        else:
            ax = plt.subplot(212)
        # Get the base x and y
        all_x = np.reshape(np.linspace(self.flags.dim_x_low, 
                                        self.flags.dim_x_high, 
                                        1000, dtype=np.float), [-1,1])
        #print('shape of all_x', np.shape(all_x))
        all_y = self.simulator(self.dataset, all_x)
        #print('shape of all_y', np.shape(all_y))
        plt.plot(all_x, all_y, label='gt')
        if len(self.data_x) <= 100:
            plt.scatter(self.data_x, self.data_y,s=5)
        # Plot the predicted value and the uncertainty
        all_yp = self.ensemble_predict_mat(all_x)           # Get the matrix format of all_yp
        # plot each individual curves
        for i in range(len(all_yp)):
            plt.plot(all_x, all_yp[i], alpha=0.1, c='r', label='NN{}'.format(i))
        #print('shape of all_yp', np.shape(all_yp))
        avg_y = np.mean(all_yp, axis=0)                     # Get the average y
        #print('shape of avg_y', np.shape(avg_y))
        std_y = np.sqrt(np.var(all_yp, axis=0))             # Get the variance
        #print('MSE of all y =', np.mean(MSE(avg_y, all_y)))
        #print('shape of std_y', np.shape(std_y))
        plt.plot(all_x, avg_y, label='average')
        plt.plot(all_x, np.abs(all_y - avg_y), label='sqrt(MSE)')
        plt.fill_between(np.ravel(all_x), np.ravel(avg_y-std_y), np.ravel(avg_y+std_y), 
                        alpha=0.3,label='std')
        # plt.xlim([self.flags.dim_x_low[0], self.flags.dim_x_high[0]])
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Committee performance @ iteration {} with {} training data'.format(iteration_ind, len(self.data_x)))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if fig_ax is None:
            plt.savefig(os.path.join(save_dir, 'sine_debug_plot_@iter_{}'.format(iteration_ind)))

    def plot_meta_debug_plot(self, iteration_ind, save_dir, fig_ax=None):
        if fig_ax is None:
            f = plt.figure(figsize=[10, 3])
        else:
            ax = plt.subplot(212)
        # print('size of data x ', np.shape(self.data_x))
        # print('size of data y', np.shape(self.data_y))
        # Plot the first spectra and all the predictions
        all_y = self.data_y[0, :]
        plt.plot(self.data_y[0, :],label='gt')
        all_yp = self.ensemble_predict_mat(np.reshape(self.data_x[0, :], [1, -1]))
        # print('shape of all_yp in debugging plot', np.shape(all_yp))
        for i in range(len(all_yp)):
            plt.plot(all_yp[i, 0, :], alpha=0.1, c='r')#, label='NN{}'.format(i))
        avg_y = np.mean(all_yp, axis=0)                     # Get the average y
        #print('shape of avg_y', np.shape(avg_y))
        std_y = np.sqrt(np.var(all_yp, axis=0))             # Get the variance
        avg_y, std_y = np.ravel(avg_y), np.ravel(std_y)
        # print('shape of std y = {}, avg y = {}'.format(np.shape(std_y), np.shape(avg_y)))
        #print('MSE of all y =', np.mean(MSE(avg_y, all_y)))
        #print('shape of std_y', np.shape(std_y))
        plt.plot(avg_y, label='average')
        plt.plot(np.abs(all_y - avg_y), label='sqrt(MSE)')
        plt.fill_between(np.arange(len(avg_y)), np.ravel(avg_y-std_y), np.ravel(avg_y+std_y), 
                        alpha=0.3,label='std')
        # plt.xlim([self.flags.dim_x_low[0], self.flags.dim_x_high[0]])
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if fig_ax is None:
            plt.savefig(os.path.join(save_dir, 'meta_debug_plot_@iter_{}'.format(iteration_ind)))


    def peak_spectra_meta(self, peak_num=10, save_dir='debug'):
        """
        The function to take a peak at the spectra to make sure they are fine
        """
        if not self.flags.data_set in ['ADM','Shell','Stack']:
            print('Peak spectra should be a meta-material dataset')
            return 
        # Peak the training set
        f = plt.figure()
        for i in range(peak_num):
            plt.plot(self.data_y[i, :])
        plt.savefig(os.path.join(save_dir, 'peak_train_spectra.png'))
        # Peak the test set
        f = plt.figure()
        for i in range(peak_num):
            plt.plot(self.test_Y[i, :])
        plt.savefig(os.path.join(save_dir, 'peak_test_spectra.png'))


    def plot_both_plots(self, iteration_ind, save_dir='results/fig'):
        #print('plotting debugging plots!')
        f = plt.figure(figsize=[10, 6])
        if 'sin' in self.flags.data_set:
            self.get_training_data_distribution(iteration_ind=iteration_ind, save_dir=save_dir, fig_ax=f)
            self.plot_sine_debug_plot(iteration_ind=iteration_ind, save_dir=save_dir, fig_ax=f)
            f.savefig(os.path.join(save_dir, 'both_plot_@iter_{}'.format(iteration_ind)))
        elif self.flags.data_set in ['Stack', 'ADM']:
            self.plot_meta_debug_plot(iteration_ind=iteration_ind, save_dir=save_dir, fig_ax=f)
            f.savefig(os.path.join(save_dir, 'both_plot_@iter_{}'.format(iteration_ind)))
        plt.cla()

    def plot_train_loss_tracker(self, save_dir='results/fig'):
        """
        Debugging for 11.06
        """
        f = plt.figure(figsize=[10, 6])
        ax = plt.subplot(211)
        if self.naal:
            loop_num = 1
        else:
            loop_num = self.n_model
        for i in range(loop_num):
            plt.plot(self.train_loss_tracker[i], 'b-', alpha=0.5)
            plt.plot(self.test_loss_tracker[i], 'r--', alpha=0.5)
        plt.yscale('log')
        y_max_list, y_min_list = [], []
        for i in range(loop_num):
            y_max_list.append(np.max(self.train_loss_tracker[i]))
            y_min_list.append(np.min(self.train_loss_tracker[i]))
        y_max = max(y_max_list)
        y_min = min(y_min_list)
        #ax = plt.subplot(212)
        data_num = np.array(range(self.flags.al_n_step)) * self.flags.al_n_dx + self.flags.al_n_x0
        step_num = np.array(range(len(data_num))) * self.flags.train_step
        for i in range(len(step_num)):
            plt.plot([step_num[i], step_num[i]], [y_min, y_max],'--',label='num={}'.format(data_num[i]), alpha=0.1)
        plt.legend()
        #plt.plot(self.train_loss_tacker_epoch[0])
        plt.title('bs_{}_retrain_{}'.format(self.flags.batch_size, self.flags.reset_weight))
        f.savefig(os.path.join(save_dir, 'loss_tracker.png'), transparent=True)

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
        
