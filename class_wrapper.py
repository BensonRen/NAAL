"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys

from torch._C import dtype
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

def MSE(pred, truth):
    """
    The MSE numpy version
    :param: pred/truth [#data, #dim]
    :return: list of MSE with len = #data
    """
    return np.mean(np.square(truth - pred), axis=1)

class Network(object):
    def __init__(self, model_fn, flags, ckpt_dir=os.path.join(os.path.abspath(''), 'models')):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs    
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

        self.models = []
        # Creating the Committee
        for i in range(flags.al_n_model):
            self.models.append(self.create_model())                        # The model itself
        self.print_model_stats()                                # Print the stats

        # Setting up the simulator
        self.simulator = self.init_simulator(self.flags.data_set)

        # Creating the training dataset
        if flags.load_dataset is None:
            self.data_x, self.data_y = self.init_dataset(self.flags.al_n_x0)
        else:
            self.data_x, self.data_y =  self.load_dataset(flags.load_dataset)
        
        # Creating the random validation dataset
        self.val_x, self.val_y = self.init_dataset(self.flags.al_test_n)        # Currently the validation set is the same size as test set

        # Creat the test set
        if flags.load_testset is None:
            self.test_X , self.test_Y = self.init_dataset(flags.al_test_n)
        else:
            self.test_X , self.test_Y = self.load_dataset(flags.load_testset)
        #print('dtype of train x', self.data_x.dtype)
        #print('dtype of train y', self.data_y.dtype)
        print('finish initializaiton')

    def print_model_stats(self):
        """
        print the model statistics and total number of trainable paramter
        """
        model = self.models[0]
        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Number of Parameters: {}".format(pytorch_total_params))

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
        data_y = self.simulator(data_x)
        return data_x, data_y

    def save_dataset(self, save_name=None, testset=False):
        """
        The function to save the dataset
        """
        # Make sure the dataset save to somewhere
        if save_name is None:
            save_name = os.path.join('dataset_saved', time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        if testset:
            np.save(os.path.join(save_name, 'test', 'data_x.npy'), self.test_x)
            np.save(os.path.join(save_name, 'test', 'data_y.npy'), self.test_y)
        else:
            np.save(os.path.join(save_name, 'data_x.npy'), self.data_x)
            np.save(os.path.join(save_name, 'data_y.npy'), self.data_y)
        print("Your dataset has been saved to ", save_name)

    def load_dataset(self, load_dataset):
        """
        The function to load the dataset
        """
        return np.load(os.path.join(load_dataset, 'data_x.npy')),  np.load(os.path.join(load_dataset, 'data_y.npy'))

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_size=(128, 8))
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
        data = DataSetClass(data_x, data_y)
        return torch.utils.data.DataLoader(data, batch_size=self.flags.batch_size)

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

    def train_single(self, model_ind, verbose=False):
        """
        The major training function. This would start the training using information given in the flags
        :param model_ind: The index of the model that would like to train
        :return: None
        """
        best_validation_loss = inf

        # Get the train loader from the data_x data_y
        train_loader = self.get_loader(self.data_x, self.data_y)
        val_loader = self.get_loader(self.val_x, self.val_y)

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.models[model_ind].cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer(model_ind)
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            self.models[model_ind].train()
            for j, (X, Y) in enumerate(train_loader):
                if cuda:
                    X = X.cuda()                                    # Put data onto GPU
                    Y = Y.cuda()                                    # Put data onto GPU
                self.optm.zero_grad()                               # Zero the gradient first
                logit = self.models[model_ind](X.float())                    # Get the output
                loss = self.make_loss(logit, Y)                     # Get the loss tensor
                loss.backward()                                     # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                train_loss += loss                                  # Aggregate the loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)

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
                
                if verbose:
                    print("For model %d, this is Epoch %d, training loss %.5f, validation loss %.5f" \
                        % (model_ind, epoch, train_avg_loss, test_avg_loss ))

                # Model improving, save the model down
                if test_avg_loss < best_validation_loss:
                    best_validation_loss = test_avg_loss
                    self.save_single(model_ind)
                    #print("Saving the model down...")

                    if best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, best_validation_loss))
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)

    def train(self):
        """
        Aggregate function of the training all models
        """
        for i in range(self.n_model):
            #print('training model ', i)
            self.train_single(i)

    def eval_model(self, model_ind, eval_X, eval_Y):
        """
        Evaluation of 
        """
        print('doing evaluationg on model ', model_ind )
        Ypred = self.pred(model_ind, eval_X, output_numpy=True)
        mse = MSE(Ypred, eval_Y)
        print('model {} has mse = {}'.format(model_ind, np.mean(mse)))
        return mse

    def pred_model(self, model_ind, test_X, output_numpy=False):
        """
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

        # Start predicting
        self.models[model_ind].eval()
        Ypred = self.models[model_ind](test_X.float())

        # Convert to numpy if necessary
        if output_numpy:
            Ypred = Ypred.cpu().data.numpy()
        return Ypred

    def ensemble_predict_mat(self, test_X):
        """
        Get each model to predict and output a large matrix
        """
        Ypred_mat = np.zeros([self.n_model, len(test_X), self.flags.dim_y])
        for i in range(self.n_model):
            Yp = self.pred_model(i, test_X, output_numpy=True)
            Ypred_mat[i, :, :] = Yp
        return Ypred_mat

    def ensemble_predict(self, test_X):
        """
        Get the average output Y for a given dataset test_X
        """
        Ypred_mat = self.ensemble_predict_mat(test_X)       # Get the Ypred mat from each model prediction
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
        """
        Ypred_mat = self.ensemble_predict_mat(test_X)
        mean_pred = np.mean(Ypred_mat, axis=0)
        var = np.mean(np.mean(np.square(Ypred_mat - mean_pred),axis=0), axis=-1)
        print('the shape of the variance output is ', np.shape(var))
        return var

    def add_X_into_trainset(self, additional_X, additional_Y=None):
        """
        Add the additional_X (optinal additional_Y) into the training set to self.data_x and self.data_y
        They are all in numpy format
        """
        # Simulate Y if it is not provided
        if additional_Y is None:
            additional_Y = self.simulator(additional_X)
        self.data_x = np.concatenate([self.data_x, additional_X])
        self.data_y = np.concatenate([self.data_y, additional_Y])

    def get_additional_X(self):
        """
        Select the additional X from a pool (that is randomly generated)
        """
        pool_x = self.random_sample_X(self.flags.al_x_pool)                 # Generate some random samples for making the pool
        pool_y = self.simulator(pool_x)
        pool_x_pred_y = self.ensemble_predict(pool_x)    # make ensemble predictions
        if self.flags.al_mode == 'MSE':
            pool_mse = MSE(pool_x_pred_y, pool_y)                               # rank the ensembled prediction and get the top ones 
            index = np.argsort(pool_mse)
        elif self.flags.al_mode == 'VAR':
            pool_VAR = self.ensemble_VAR(pool_x)
            index = np.argsort(pool_VAR)
        elif self.flags.al_mode == 'Random':
            index = np.random.permutation(len(pool_x))
        return pool_x[index[-self.flags.al_n_dx:]]
    
    def active_learn(self):
        """
        The main active learning function
        """
        # First start from a base model of training 
        self.train()
        print('finish model initial training')
        mse = np.mean(self.ensemble_MSE(self.test_X, self.test_Y))
        print('Initial ensemble mse = ', mse)
        # Active learning part
        for al_step in range(self.flags.al_n_step):
            # First we select the additional X
            additional_X = self.get_additional_X()
            # Put them into training set
            self.add_X_into_trainset(additional_X)
            # Train again here
            self.train()
            # Calculate mse and report that
            mse = np.mean(self.ensemble_MSE(self.test_X, self.test_Y))
            print('AL step {}, current train set size = {}, ensemble mse = {}, the AL_mode is {}'.format(al_step, 
                    len(self.data_x), mse, self.flags.al_mode))

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
        