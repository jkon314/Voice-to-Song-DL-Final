import torch
import torch.nn as nn





    




class styleEncoder(nn.module):


    def __init__(self, in_size, out_size,hidden_size,device):
        self.in_size = in_size
        self.hidden_size = hidden_size


        
        self.lstm = nn.LSTM(self.in_size,self.hidden_size,2,device=device)
        self.fc   = nn.Linear(self.hidden_size,out_size,device=device)


    def forward(self, input):
        

        out = self.lstm(input)
        out = self.fc(out)

        return out

class Encoder(nn.module):

    """
    Encoder which will apply convolutional layers to the output of the autoencoder

    Parameters: 
    in_channels: number of channels in audio input
    k_size: kernel window size to use for 1-D convolution
    out_channels, number of kernels to use/ number of output channels generated by convolution should be 512 per paper
    hid_size: hidden size to be used by BLSTM Layer
    """

    def __init__(self,in_channels, speaker_inSize, singer_inSize, k_size, out_channels, hidden_size,device,full_size,out_size,down_sample_factor):


        self.k_size = k_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.singer_inSize = singer_inSize
        self.speaker_inSize = speaker_inSize
        self.hidden_size = hidden_size
        self.device = device
        self.full_size = full_size
        self.out_size = out_size
        self.convs = nn.ModuleList()
        self.downsamplefactor = down_sample_factor
        for c in range(0,3):

            if(c == 0):
                newInChannels = self.in_channels #first convolution layer will have different dimension
            else:
                newInChannels = self.out_channels
            self.convs.append(
                
                nn.Sequential(
                    nn.Conv1d(newInChannels,self.out_channels,self.k_size,1,0,1,device=self.device),
                    nn.BatchNorm1d(self.out_channels,device=self.device),
                    nn.ReLU(device=self.device)

                )


            )

        self.blstm = nn.LSTM(self.out_channels,self.hidden_size,2,batch_first=True,bidirectional=True)

         






        #initialize style encoder components to run on content to encode (X2 or lower Branch)
        
        
    def forward(self,spec,style):

        """
        Performs encoder foward pass
        Parameters: spec: spectrogram of singer input
                    style: resembylzer style encoding of speaker?
        Returns: output: resulting encoded vocals
        """
        

        

        concat = torch.cat((spec,style))

        output = self.convs(concat)
        
        output, cell = self.blstm(output) #do not use cell in this scenario

        fwd , back = torch.split(output,output.shape[2]//2,dim=2) #split into forward and backward components of 


        fwd = torch.nn.functional.interpolate(fwd,None,(1,1,self.downsamplefactor),'linear')
        back = torch.nn.functional.interpolate(back,None,(1,1,self.downsamplefactor),'linear')

        




        return fwd,back
    

        
