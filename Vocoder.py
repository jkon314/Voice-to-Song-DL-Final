import torch
import numpy as np
import os
import soundfile as sf

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
        N, freq_bins, time_bins = magnitude_spectrogram.shape
        # Initialize phase randomly
        angle = 2 * np.pi * torch.rand(magnitude_spectrogram.size()).to(magnitude_spectrogram.device)
        stft_matrix = magnitude_spectrogram * torch.exp(1j * angle)
        
        for _ in range(self.num_iters):
            # Perform inverse STFT to get the time-domain signal for each sample
            signal_batch = []
            for i in range(N):
                signal = self.istft(stft_matrix[i])
                signal_batch.append(signal)
            signal_batch = torch.stack(signal_batch)
            
            # Perform STFT to get the spectrogram for each sample
            stft_matrix_batch = []
            for i in range(N):
                stft_matrix = self.stft(signal_batch[i])
                stft_matrix_batch.append(stft_matrix)
            stft_matrix_batch = torch.stack(stft_matrix_batch)
            
            # Update the magnitude with the original magnitude
            stft_matrix = magnitude_spectrogram * torch.exp(1j * torch.angle(stft_matrix_batch))
        
        # Get the final reconstructed signal for each sample
        final_signals = []
        for i in range(N):
            signal = self.istft(stft_matrix[i])
            final_signals.append(signal)
        final_signals = torch.stack(final_signals)
        return final_signals

    def reconstruct(self, magnitude_spectrogram, output_dir, sample_rate=16000):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        reconstructed_signals = self.griffin_lim(magnitude_spectrogram)
        
        # Save each reconstructed waveform as a WAV file
        for i, signal in enumerate(reconstructed_signals):
            output_path = os.path.join(output_dir, f"reconstructed_{i}.wav")
            # soundfile.write expects the signal to be a numpy array
            sf.write(output_path, signal.cpu().numpy(), sample_rate)

        return reconstructed_signals