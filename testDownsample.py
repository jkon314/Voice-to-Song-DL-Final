import torch
import torch.nn as nn


print(torch.nn.functional.interpolate(torch.tensor([[[1.1,2.1],[3.1,4.1]],[[1.1,2.1],[3.1,4.1]]]),None,0.5,'nearest'))