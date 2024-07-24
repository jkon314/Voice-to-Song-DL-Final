# Class 'Vocoder', which includes methods for performing the STFT, ISTFT, 
# and Griffin-Lim algorithm, as well as a method to reconstruct 
# the time-domain signal from a spectrogram.

import torch
import numpy as np

class Vocoder:
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, num_iters=30):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_iters = num_iters
        self.window = torch.hann_window(win_length)

    def stft(self, signal):
        return torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, 
                          win_length=self.win_length, window=self.window, return_complex=True)

    def istft(self, stft_matrix):
        return torch.istft(stft_matrix, n_fft=self.n_fft, hop_length=self.hop_length, 
                           win_length=self.win_length, window=self.window)

    # The Griffin-Lim Method
    def griffin_lim(self, magnitude_spectrogram): 
        # Initialize phase randomly
        angle = 2 * np.pi * torch.rand(magnitude_spectrogram.size()).to(magnitude_spectrogram.device)
        stft_matrix = magnitude_spectrogram * torch.exp(1j * angle)
        
        for _ in range(self.num_iters):
            # Perform inverse STFT to get the time-domain signal
            signal = self.istft(stft_matrix)
            
            # Perform STFT to get the spectrogram
            stft_matrix = self.stft(signal)
            
            # Update the magnitude with the original magnitude
            stft_matrix = magnitude_spectrogram * torch.exp(1j * torch.angle(stft_matrix))
        
        # Get the final reconstructed signal
        signal = self.istft(stft_matrix)
        return signal

    def reconstruct(self, magnitude_spectrogram):
        return self.griffin_lim(magnitude_spectrogram)