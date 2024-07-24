import random

import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_activation_function


class Decoder(nn.Module):
    """ The Decoder module 
    """

    def __init__(
            self, 
            input_size=320, 
            cnn_num_layers = 3,
            cnn_output_channels=512, 
            cnn_activation_function="relu",
            cnn_kernel_size=5,
            cnn_stride=1,
            cnn_padding=2,
            lstm_input_size=512,
            lstm_hidden_size=2048,
            lstm_num_layers=2,
            lstm_batch_first=True,
            lstm_bidirectional=True,
            output_size=80):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.cnn_num_layers = cnn_num_layers
        self.cnn_output_channels = cnn_output_channels
        self.cnn_activation_function = cnn_activation_function
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_stride = cnn_stride
        self.cnn_padding = cnn_padding
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_batch_first = lstm_batch_first
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.output_size = output_size

        # CNN Set 1
        self.cnn_set = []
        self.cnn_set.append(nn.Sequential(
            nn.Conv1d(
                in_channels=self.lstm_hidden_size * 2 if self.lstm_bidirectional else self.lstm_hidden_size, 
                out_channels=self.cnn_output_channels,
                kernel_size=self.cnn_kernel_size,
                stride=self.cnn_stride,
                padding=self.cnn_padding,
            ),
            nn.BatchNorm1d(num_features=self.cnn_output_channels)
        ))

        for _ in range(self.cnn_num_layers - 1):
            self.cnn_set.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=self.cnn_output_channels, 
                    out_channels=self.cnn_output_channels,
                    kernel_size=self.cnn_kernel_size,
                    stride=self.cnn_stride,
                    padding=self.cnn_padding,
                ),
                nn.BatchNorm1d(num_features=self.cnn_output_channels)
            ))

        self.cnn_set = nn.ModuleList(self.cnn_set)

        # LSTM Layers
        self.lstm1 = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.lstm_hidden_size, 
            num_layers=1, 
            batch_first=self.lstm_batch_first,
            bidirectional=self.lstm_bidirectional
        )

        self.lstm2 = nn.LSTM(
            input_size=self.lstm_input_size, 
            hidden_size=self.lstm_hidden_size, 
            num_layers=self.lstm_num_layers - 1, 
            batch_first=self.lstm_batch_first,
            bidirectional=self.lstm_bidirectional
        )

        # Linear 1x1
        self.linear = nn.Linear(
            in_features=self.lstm_hidden_size * 2 if self.lstm_bidirectional else self.lstm_hidden_size, 
            out_features=self.output_size
        )
    
        self.cnn_activation = get_activation_function(self.cnn_activation_function)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
        """
        outputs = input # (N, Input Channels, T)

        # LSTM 1
        outputs = outputs.transpose(1, 2) # (N, T, Input Channels)
        outputs, _= self.lstm1(outputs) # (N, T, LSTM Hidden Size)
        outputs = outputs.transpose(1, 2) # (N, LSTM Hidden Size, T)

        # CNN Set 1
        for i in range(self.cnn_num_layers):
            outputs = self.cnn_activation(self.cnn_set[i](outputs)) # (N, CNN Set 1 Output Channels, 1)

        # LSTM 2
        outputs = outputs.transpose(1, 2) # (N, T, Output Channels)
        outputs, _= self.lstm2(outputs) # (N, T, LSTM Hidden Size)

        # Linear
        outputs = self.linear(outputs) # (N, T, Output Size)
        outputs = outputs.transpose(1, 2) # (N, Output Size, T)
    
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs
