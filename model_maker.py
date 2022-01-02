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

        # Conv Layer definitions here
        self.convs = nn.ModuleList([])
        in_channel = 1                                                  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel,
                                                                     flags.conv_kernel_size,
                                                                     flags.conv_stride)):
            if stride == 2:     # We want to double the number
                pad = int(kernel_size/2 - 1)
            elif stride == 1:   # We want to keep the number unchanged
                pad = int((kernel_size - 1)/2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")

            self.convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                stride=stride, padding=pad)) # To make sure L_out double each time
            in_channel = out_channel # Update the out_channel
        if len(self.convs):                     # If there are upconvolutions, do the convolution back to single channel
            self.convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

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
        
        out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
        # For the conv part
        for ind, conv in enumerate(self.convs):
            #print(out.size())
            out = conv(out)
        S = out.squeeze(1)
        return S

class Dropout_model(nn.Module):
    def __init__(self, flags):
        super(Dropout_model, self).__init__()
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        # For dropout layer we would not use the batchnorm, as this is not working!!!
        self.bn_linears = nn.ModuleList([])
        self.dropouts = nn.Dropout(0.5)#nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))#, momentum=0.01))
            #self.dropouts.append()
            # self.bn_linears.append(nn.LayerNorm(flags.linear[ind + 1]))

        # Conv Layer definitions here
        self.convs = nn.ModuleList([])
        in_channel = 1                                                  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel,
                                                                     flags.conv_kernel_size,
                                                                     flags.conv_stride)):
            if stride == 2:     # We want to double the number
                pad = int(kernel_size/2 - 1)
            elif stride == 1:   # We want to keep the number unchanged
                pad = int((kernel_size - 1)/2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")

            self.convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                stride=stride, padding=pad)) # To make sure L_out double each time
            in_channel = out_channel # Update the out_channel
        if len(self.convs):                     # If there are upconvolutions, do the convolution back to single channel
            self.convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

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
        # for ind, fc in enumerate(self.linears):
            # print(out)
            # print(out.size())
            if ind != len(self.linears) - 1:
                out = F.leaky_relu(fc(out))                                   # ReLU + BN + Linear
                # print(out)
                # out = F.leaky_relu(bn(fc(out)))                                   # ReLU + BN + Linear
                if ind == len(self.linears) - 2:
                # if ind == len(self.linears) - 4:
                    out = self.dropouts(out)
                # out = F.relu(fc(out))
                # out = F.leaky_relu(bn(fc(out)))       
                # out = F.leaky_relu(fc(out))    
                #print('out size', out.size())
            else:
                out = fc(out)                                           # For last layer, no activation function
        
        # print(out)
        # print(out.size())
        # quit()

        out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
        # For the conv part
        for ind, conv in enumerate(self.convs):
            #print(out.size())
            out = conv(out)
        S = out.squeeze(1)
        return S
        
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
        