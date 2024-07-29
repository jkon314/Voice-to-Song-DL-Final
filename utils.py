import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import librosa

import matplotlib.pyplot as plt

from pydub import AudioSegment
from pydub.playback import play


def get_activation_function(activation_function):
    if activation_function == "relu":
        return nn.ReLU()
    elif activation_function == "tanh":
        return nn.Tanh()
    elif activation_function == "softmax":
        return nn.Softmax()
    else:
        return nn.ReLU()
    
def criterion(input, encoder_ouput, decoder_output, postnet_output, encoded_postnet_output, style_loss, style_loss_param=10000, gamma=10000): 
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    L1 = mse_loss(input, decoder_output)
    L2 = mse_loss(input, postnet_output)
    L3 = l1_loss(encoder_ouput, encoded_postnet_output)
    
    return L1 + L2 + (style_loss / style_loss_param), np.array([L1.item(), L2.item(), L3.item() * gamma, style_loss.item() / style_loss_param])

def overlay_audio(file1, file2, export_location="./", export_format="mp3", position=0):
    sound1 = AudioSegment.from_wav(file1) - 3
    sound2 = AudioSegment.from_mp3(file2) + 6

    overlay = sound2.overlay(sound1, position=position)
    overlay.export(export_location, format="mp3")


def combine_spec_and_style(spec,style):
    if len(style.shape) == 1:
        style = style.unsqueeze(0)
    copy = style[:,:,None] #add additional dimension to duplicate style embeddings over

    style_dup = torch.repeat_interleave(copy,spec.shape[2],2)
    
    # print(spec.shape)
    # print(style_dup.shape)
    concat = torch.cat((spec,style_dup),1) #concatenate the spectrogram and style to create Nx512xT tensor

    return concat


def train(model, randomCNN, a_S, dataloader, optimizer, criterion, device='cpu', style_loss_param=10000):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", ascii=True)
    total_losses = np.zeros(4)

    for batch_idx, data in enumerate(progress_bar):
        singing_spec, speech_style = data
        singing_spec, speech_style = singing_spec.to(device), speech_style.to(device)
        

        optimizer.zero_grad()
        spec, encoder_out, decoder_out, postnet_out, encoded_postnet_out = model((singing_spec, speech_style))
        
        a_G = randomCNN(decoder_out.unsqueeze(0))
        
        style_loss = 1 * compute_layer_style_loss(a_S, a_G)
        
        loss, losses = criterion(spec, encoder_out, decoder_out, postnet_out, encoded_postnet_out, style_loss, style_loss_param=style_loss_param) #edited to unpack multiple returns
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_losses += losses
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    avg_total_losses = total_losses / len(dataloader)
    return total_loss, avg_loss, avg_total_losses


# def evaluate(model, dataloader, criterion, device='cpu'):
#     model.eval()
#     total_loss = 0.0
#     total_losses = np.zeros(3)
#     with torch.no_grad():
#         progress_bar = tqdm(dataloader, desc="Evaluating", ascii=True)

#         for batch_idx, data in enumerate(progress_bar):
#             singing_spec, speech_style = data
#             singing_spec, speech_style = singing_spec.to(device), speech_style.to(device)
#             spec, encoder_out, decoder_out, postnet_out, encoded_postnet_out = model((singing_spec, speech_style))
#             loss, losses = criterion(spec, encoder_out, decoder_out, postnet_out, encoded_postnet_out) #edited to unpack multiple returns
#             total_loss += loss.item()
#             total_losses += losses
#             progress_bar.set_postfix(loss=loss.item())

#     avg_loss = total_loss / len(dataloader)
#     avg_total_losses = total_losses / len(dataloader)
#     return total_loss, avg_loss, avg_total_losses

# Example usage:
# model = Model()
# dataloader = DataLoader(your_dataset, batch_size=32, shuffle=False)
# criterion = nn.MSELoss()  # or whatever loss function is appropriate for your task
# total_loss, avg_loss = evaluate(model, dataloader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Total Loss: {total_loss:.4f}, Average Loss: {avg_loss:.4f}')


def plot_curves(train_loss_history, filename):
    '''
    Plot learning curves with matplotlib. Training loss and validation loss are plotted in the same figure.
    :param train_loss_history: training loss history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param filename: filename for saving the plot
    :return: None, save plot in the current directory
    '''
    epochs = range(len(train_loss_history))
    plt.plot(epochs, train_loss_history, label='Train Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves - ' + filename)
    plt.savefig(filename + '.png')
    plt.show()

    
def gram(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_L)

    Returns:
    GA -- Gram matrix of shape (n_C, n_C)
    """
    GA = torch.matmul(A, A.t())

    return GA


def gram_over_time_axis(A):
    """
    Argument:
    A -- matrix of shape (1, n_C, n_H, n_W)

    Returns:
    GA -- Gram matrix of A along time axis, of shape (n_C, n_C)
    """
    m, n_C, n_H, n_W = A.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    A_unrolled = A.view(m * n_C * n_H, n_W)
    GA = torch.matmul(A_unrolled, A_unrolled.t())

    return GA


def compute_layer_style_loss(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)

    Returns:
    J_style_layer -- tensor representing a scalar style cost.
    """
    m, n_C, n_H, n_W = a_G.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)

    # Calculate the gram
    # !!!!!! IMPORTANT !!!!! Here we compute the Gram along n_C,
    # not along n_H * n_W. But is the result the same? No.
    GS = gram_over_time_axis(a_S)
    GG = gram_over_time_axis(a_G)

    # Computing the loss
    J_style_layer = 1.0 / (4 * (n_C ** 2) * (n_H * n_W)) * torch.sum((GS - GG) ** 2)

    return J_style_layer
