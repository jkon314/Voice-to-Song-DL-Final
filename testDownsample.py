import torch
import torch.nn as nn
import torchaudio as ta


vocals = ta.load('collardgreens.mp3')



transform = ta.transforms.Spectrogram(510)

out = transform(vocals[0])

print(out.shape)