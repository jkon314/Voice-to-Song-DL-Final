import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

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


def train(model, dataloader, optimizer, criterion, scheduler=None, device='cpu', epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', ascii=True)
        
        for batch_idx, data in enumerate(progress_bar):
            # Assuming data is a tuple of (singing_spec, speech_style, target)
            singing_spec, speech_style, target = data
            singing_spec, speech_style, target = singing_spec.to(device), speech_style.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model((singing_spec, speech_style))

            # Compute the loss
            loss = criterion(output, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Learning rate scheduler step
            if scheduler:
                scheduler.step()
            
            # Accumulate the loss
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

    return model

# Example usage:
# model = Model()
# dataloader = DataLoader(your_dataset, batch_size=32, shuffle=True)
# optimizer = Adam(model.parameters(), lr=1e-3)
# criterion = nn.MSELoss()  # or whatever loss function is appropriate for your task
# train(model, dataloader, optimizer, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, dataloader, criterion, device='cpu'):
    # Set the model to evaluation mode to avoid updating weights
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        # Get the progress bar for evaluation
        progress_bar = tqdm(dataloader, desc="Evaluating", ascii=True)

        for batch_idx, data in enumerate(progress_bar):
            # Assuming data is a tuple of (singing_spec, speech_style, target)
            singing_spec, speech_style, target = data
            singing_spec, speech_style, target = singing_spec.to(device), speech_style.to(device), target.to(device)

            # Forward pass
            output = model((singing_spec, speech_style))

            # Compute the loss
            loss = criterion(output, target)

            # Accumulate the loss
            total_loss += loss.item()

            # Update the progress bar with the current batch loss
            progress_bar.set_postfix(loss=loss.item())

    # Calculate the average loss over the entire dataset
    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss

# Example usage:
# model = Model()
# dataloader = DataLoader(your_dataset, batch_size=32, shuffle=False)
# criterion = nn.MSELoss()  # or whatever loss function is appropriate for your task
# total_loss, avg_loss = evaluate(model, dataloader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Total Loss: {total_loss:.4f}, Average Loss: {avg_loss:.4f}')