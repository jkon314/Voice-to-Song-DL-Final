import torch
import numpy as np

class preppedData(torch.utils.data.Dataset):

    def __init__(self,specPath,stylePath):

        self.spec_list = np.load(specPath)
        self.style_list = np.load(stylePath)

    def __getitem__(self,idx):

        spec = self.spec_list[idx]
        style = self.style_list[idx]
        return spec,style

    def __len__(self):
        #since the datasets are incongruous do this idk 
        return min(self.spec_list.shape[0],self.style_list.shape[0])