"""
S2S Decoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_activation_function

class PostNet(nn.Module):
    """ The Decoder module 
    """

    def __init__(
            self, 
            input_size=80, 
            num_layers = 5,
            hidden_size=512,
            activation_function="tanh",
            cnn_kernel_size=5,
            cnn_stride=1,
            cnn_padding=2,
            output_size=80):
        super(PostNet, self).__init__()

        self.num_layers =num_layers
        self.hidden_size =hidden_size
        self.activation_function =activation_function
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_stride = cnn_stride
        self.cnn_padding = cnn_padding
        self.input_size = input_size
        self.output_size = output_size


        # CNN Set 2 (PostNet)
        self.cnn_set = []
        self.cnn_set.append(nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_size, 
                out_channels=self.hidden_size,
                kernel_size=self.cnn_kernel_size,
                stride=self.cnn_stride,
                padding=self.cnn_padding,
            ),
            nn.BatchNorm1d(self.hidden_size)
        ))

        for _ in range(self.num_layers - 2):
            self.cnn_set.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=self.hidden_size, 
                    out_channels=self.hidden_size,
                    kernel_size=self.cnn_kernel_size,
                    stride=self.cnn_stride,
                    padding=self.cnn_padding,
                ),
                nn.BatchNorm1d(self.hidden_size)
            ))

        self.cnn_set.append(nn.Conv1d(
            in_channels=self.hidden_size, 
            out_channels=self.output_size,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
            padding=self.cnn_padding,
        ))

        self.cnn_set = nn.ModuleList(self.cnn_set)
        
        self.activation = get_activation_function(self.activation_function)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, Input Size, 1). HINT: encoded does not mean from encoder!!
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
        """
        outputs = input # (N, Input Size, T)
        
        # CNN Set 2 (PostNet)
        for i in range(self.num_layers - 1):
            outputs = self.activation(self.cnn_set[i](outputs)) # (N, Hidden Size, T)

        # Don't use tanh on last layer
        outputs = self.cnn_set[-1](outputs) # (N, Output Size, T)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs
