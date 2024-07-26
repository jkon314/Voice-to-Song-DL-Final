import torch
import torch.nn as nn
import torchaudio as ta


vocals = ta.load('Tracks/vocals_chunks/001  Iggy Azalea - Fancy (Feat. Charli XCX)_(Vocals)_UVR-MDX-NET-Inst_HQ_3-chunk-1.mp3')



transform = ta.transforms.Spectrogram(510)
print(vocals[0].shape)
out = transform(vocals[0])

print(out.shape)