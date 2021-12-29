"""
This is the module where the model is defined. It uses the nn.Module as backbone to create the network structure
"""
# Own modules

# Built in
import math
# Libs
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import pow, add, mul, div, sqrt


class NN(nn.Module):
    def __init__(self, flags):
        super(NN, self).__init__()
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))#, momentum=0.01))
            # self.bn_linears.append(nn.LayerNorm(flags.linear[ind + 1]))

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out
        #print('G size', G.size())
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if ind != len(self.linears) - 1:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
                # out = F.relu(fc(out))
                # out = F.leaky_relu(bn(fc(out)))       
                # out = F.leaky_relu(fc(out))    
                #print('out size', out.size())
            else:
                out = fc(out)                                           # For last layer, no activation function
        return out

class Dropout(nn.Module):
    def __init__(self, flags):
        super(Dropout, self).__init__()
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        self.dropouts = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))#, momentum=0.01))
            self.dropouts.append(nn.Dropout(0.5))
            # self.bn_linears.append(nn.LayerNorm(flags.linear[ind + 1]))

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out
        #print('G size', G.size())
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if ind != len(self.linears) - 1:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
                out = self.dropouts[ind](out)
                # out = F.relu(fc(out))
                # out = F.leaky_relu(bn(fc(out)))       
                # out = F.leaky_relu(fc(out))    
                #print('out size', out.size())
            else:
                out = fc(out)                                           # For last layer, no activation function
        return out

class NAAL(nn.Module):
    """
    The ensemble model of a number of networks
    """
    def __init__(self, flags):
        super(NAAL, self).__init__()
        self.sub_model_list = nn.ModuleList([])
        self.nmod = flags.al_n_model
        for i in range(self.nmod):
            self.sub_model_list.append(NN(flags))
        print('self sub_model len', len(self.sub_model_list))
    
    # def __getitem__(self, index):
    #     """
    #     The list like structure
    #     """
    #     if index >= 0:
    #         return self.sub_model_list[index]
    #     elif index == -1:
    #         return self

    def forward(self, G):
        """
        The forward model that takes all the models
        """
        #output_mat = torch.zeros([self.nmod, *G.size()], device='cuda')
        output_list = []
        for ind, model in enumerate(self.sub_model_list):
            output_list.append(model(G))
            #output_mat[ind, :, :] = model(G)
        return torch.stack(output_list)
        #return output_mat
        