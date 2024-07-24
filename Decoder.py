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


class Decoder(nn.Module):
    """ The Decoder module 
    """

    def __init__(
            self, 
            cnn_set1_input_channels=320, 
            cnn_set1_output_channels=512, 
            cnn_set3_output_channels=512,
            cnn_set4_input_channels=512,
            cnn_kernel_size=5,
            cnn_stride=1,
            cnn_padding=2,
            lstm_input_size=512,
            lstm_hidden_size=1024,
            lstm_num_layers=3,
            lstm_batch_first=True,
            output_size=80):
        super(Decoder, self).__init__()

        self.cnn_set1_input_channels = cnn_set1_input_channels
        self.cnn_set1_output_channels = cnn_set1_output_channels
        self.cnn_set3_output_channels = cnn_set3_output_channels
        self.cnn_set4_input_channels = cnn_set4_input_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_stride = cnn_stride
        self.cnn_padding = cnn_padding
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_batch_first = lstm_batch_first
        self.lstm_num_layers = lstm_num_layers
        self.output_size = output_size

        # CNN Set 1
        self.conv_norm_5x1_1 = nn.Conv1d(
            in_channels=self.cnn_set1_input_channels, 
            out_channels=self.cnn_set1_output_channels,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
            padding=self.cnn_padding,
        )
        self.conv_norm_5x1_2 = nn.Conv1d(
            in_channels=self.cnn_set1_output_channels, 
            out_channels=self.cnn_set1_output_channels,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
            padding=self.cnn_padding,
        )
        self.conv_norm_5x1_3 = nn.Conv1d(
            in_channels=self.cnn_set1_output_channels, 
            out_channels=self.cnn_set1_output_channels,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
            padding=self.cnn_padding,
        )

        # LSTM Layers
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size, 
            hidden_size=self.lstm_hidden_size, 
            num_layers=self.lstm_num_layers, 
            batch_first=self.lstm_batch_first
        )

        # CNN Set 2
        self.conv_norm_1x1 = nn.Conv1d(
            in_channels=self.lstm_hidden_size,
            out_channels=self.output_size,
            kernel_size=1
        )

        # CNN Set 3
        self.conv_norm_5x1_4 = nn.Conv1d(
            in_channels=self.output_size, 
            out_channels=self.cnn_set3_output_channels,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
            padding=self.cnn_padding,
        )

        self.conv_norm_5x1_5 = nn.Conv1d(
            in_channels=self.cnn_set3_output_channels, 
            out_channels=self.cnn_set3_output_channels,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
            padding=self.cnn_padding,
        )

        self.conv_norm_5x1_6 = nn.Conv1d(
            in_channels=self.cnn_set3_output_channels, 
            out_channels=self.cnn_set3_output_channels,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
            padding=self.cnn_padding,
        )

        self.conv_norm_5x1_7 = nn.Conv1d(
            in_channels=self.cnn_set3_output_channels, 
            out_channels=self.cnn_set3_output_channels,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
            padding=self.cnn_padding,
        )

        # CNN Set 4
        self.conv_norm_5x1_8 = nn.Conv1d(
            in_channels=self.cnn_set3_output_channels, 
            out_channels=self.output_size,
            kernel_size=self.cnn_kernel_size,
            stride=self.cnn_stride,
            padding=self.cnn_padding,
        )


        self.relu = nn.ReLU()

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, input):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       If attention is true, compute the attention probabilities and use   #
        #       them to do a weighted sum on the encoder_outputs to determine       #
        #       the hidden (and cell if LSTM) states that will be consumed by the   #
        #       recurrent layer.                                                    #
        #                                                                           #
        #       Apply linear layer and log-softmax activation to output tensor      #
        #       before returning it.                                                #
        #############################################################################
        outputs = input # (N, Input Channels, 1)

        # CNN Set 1
        outputs = self.relu(self.conv_norm_5x1_1(outputs)) # (N, CNN Set 1 Output Channels, 1)
        outputs = self.relu(self.conv_norm_5x1_2(outputs)) # (N, CNN Set 1 Output Channels, 1)
        outputs = self.relu(self.conv_norm_5x1_3(outputs)) # (N, CNN Set 1 Output Channels, 1)

        # LSTM
        outputs = outputs.transpose(1, 2) # (N, 1, Output Channels)
        outputs, _= self.lstm(outputs) # (N, 1, LSTM Hidden Size)
        outputs = outputs.transpose(1, 2) # (N, LSTM Hidden Size, 1)

        # CNN Set 2
        outputs = self.conv_norm_1x1(outputs) # (N, Output Size, 1)
        temp = torch.clone(outputs) # to use for skip connection
        
        # CNN Set 3
        outputs = self.relu(self.conv_norm_5x1_4(outputs)) # (N, CNN Set 3 Output Channels, 1)
        outputs = self.relu(self.conv_norm_5x1_5(outputs)) # (N, CNN Set 3 Output Channels, 1)
        outputs = self.relu(self.conv_norm_5x1_6(outputs)) # (N, CNN Set 3 Output Channels, 1)
        outputs = self.relu(self.conv_norm_5x1_7(outputs)) # (N, CNN Set 3 Output Channels, 1)

        # CNN Set 4
        outputs = self.relu(self.conv_norm_5x1_8(outputs)) # (N, Output Size, 1)

        outputs += temp # (N, Output Size, 1)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs
