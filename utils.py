import random

import torch
import torch.nn as nn
import torch.optim as optim

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
    
def criterion(input, encoder_ouput, decoder_output, postnet_output, encoded_postnet_output, gamma=10000): 
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    L1 = mse_loss(input, decoder_output)
    L2 = mse_loss(input, postnet_output)
    L3 = l1_loss(encoder_ouput, encoded_postnet_output)
    return L1 + L2 + (gamma * L3)

def overlay_audio(file1, file2, export_location="./", export_format="mp3", position=0):
    sound1 = AudioSegment.from_wav(file1)
    sound2 = AudioSegment.from_wav(file2)

    overlay = sound1.overlay(sound2, position=position)

    play(overlay)
    overlay.export(export_location, format="mp3")


def combine_spec_and_style(spec,style):
    copy = style[:,:,None] #add additional dimension to duplicate style embeddings over

    style_dup = torch.repeat_interleave(copy,spec.shape[2],2)


    concat = torch.cat((spec,style_dup),1) #concatenate the spectrogram and style to create Nx512xT tensor

    return concat