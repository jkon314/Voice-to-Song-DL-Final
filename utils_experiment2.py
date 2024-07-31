import random
from Vocoder import Vocoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from resemblyzer import VoiceEncoder
import matplotlib.pyplot as plt
import numpy  as np
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
    
def criterion(input, encoder_ouput, decoder_output, postnet_output, encoded_postnet_output, in_style,gamma=10000): 
    mse_loss = nn.MSELoss()
    mse_loss2 = nn.MSELoss()
    mse_loss_style = nn.MSELoss()
    l1_loss = nn.L1Loss()

    #get Postnet Style
    vocoder = Vocoder(n_fft=510, hop_length=255, win_length=510, num_iters=50)
    
    
    
    detached = torch.detach(postnet_output)
    detached = detached.to('cuda')
    # Reconstruct waveforms from the batch of magnitude spectrograms
    reconstructed_waveforms = vocoder.reconstruct(detached, './', 44100,False)
    ve = VoiceEncoder(device='cuda')
    styles = []
    for wav in reconstructed_waveforms:
        wav = np.array(torch.tensor(wav,device='cpu'))
        style = ve.embed_utterance(wav)
        styles.append(style)
    

    styles = torch.tensor(styles,device='cuda')
    L1 = mse_loss(input, decoder_output)
    L2 = mse_loss2(input, postnet_output)
    L3 = l1_loss(encoder_ouput, encoded_postnet_output)
    L4 = mse_loss_style(in_style,styles)
    #print(f'L1: {L1}, L2: {L2}, L3: {L3*gamma} and L4: {L4*32000}')
    return L1 + L2 + (gamma * L3) + L4*40000

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


def train(model, dataloader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", ascii=True)

    for batch_idx, data in enumerate(progress_bar):
        #singing_spec, speech_style, target = data
        #singing_spec, speech_style, target = singing_spec.to(device), speech_style.to(device), target.to(device)

        singing_spec, speech_style, target = data
        singing_spec, speech_style, target = singing_spec.to(device), speech_style.to(device), target.to(device)

        optimizer.zero_grad()
        spec, encoder_out, decoder_out, postnet_out, encoded_postnet_out = model((singing_spec, speech_style,target))
        loss = criterion(spec, encoder_out, decoder_out, postnet_out, encoded_postnet_out,target) #edited to unpack multiple returns
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss


def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", ascii=True)

        for batch_idx, data in enumerate(progress_bar):
            singing_spec, speech_style, target = data
            singing_spec, speech_style, target = singing_spec.to(device), speech_style.to(device), target.to(device)
            spec, encoder_out, decoder_out, postnet_out, encoded_postnet_out = model((singing_spec, speech_style,target))
            loss = criterion(spec, encoder_out, decoder_out, postnet_out, encoded_postnet_out,target) #edited to unpack multiple returns
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss

# Example usage:
# model = Model()
# dataloader = DataLoader(your_dataset, batch_size=32, shuffle=False)
# criterion = nn.MSELoss()  # or whatever loss function is appropriate for your task
# total_loss, avg_loss = evaluate(model, dataloader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Total Loss: {total_loss:.4f}, Average Loss: {avg_loss:.4f}')


def plot_curves(train_loss_history, valid_loss_history, filename):
    '''
    Plot learning curves with matplotlib. Training loss and validation loss are plotted in the same figure.
    :param train_loss_history: training loss history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param filename: filename for saving the plot
    :return: None, save plot in the current directory
    '''
    epochs = range(len(train_loss_history))
    plt.plot(epochs, train_loss_history, label='Train Loss')
    plt.plot(epochs, valid_loss_history, label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves - ' + filename)
    plt.savefig(filename + '.png')
    plt.show()


