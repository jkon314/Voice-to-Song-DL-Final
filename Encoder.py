import torch
import torch.nn as nn

class styleEncoder(nn.Module):

    #alternative to resemblyzer, likely we will not implement this
    def __init__(self, in_size, out_size,hidden_size,device):
        self.in_size = in_size
        self.hidden_size = hidden_size


        
        self.lstm = nn.LSTM(self.in_size,self.hidden_size,2,device=device)
        self.fc   = nn.Linear(self.hidden_size,out_size,device=device)


    def forward(self, input):
        

        out = self.lstm(input)
        out = self.fc(out)

        return out

class Encoder(nn.Module):

    """
    Encoder which will apply convolutional layers to the output of the autoencoder

    Parameters: 
    in_channels: number of channels in audio input
    k_size: kernel window size to use for 1-D convolution
    out_channels, number of kernels to use/ number of output channels generated by convolution should be 512 per paper
    hidden_size: hidden size to be used by BLSTM Layer 
    down_sample_factor: factor by which to downsample final outputs of the encoder
    max_len: the maximum audio time length, all inputs should be padded to this length
     
    """

    def __init__(self,in_channels=512, k_size=5, out_channels=512, hidden_size=32,device='cpu',down_sample_factor=0.03125, max_len=1000, stride_size=1):

        super(Encoder, self).__init__()
        self.k_size = k_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.device = device
        self.convs = nn.ModuleList()
        self.downsamplefactor = down_sample_factor
        self.max_len = max_len
        self.stride_size = stride_size
        for c in range(0,3):

            if(c == 0):
                newInChannels = self.in_channels #first convolution layer will have potentially different input dimension
            else:
                newInChannels = self.out_channels
            self.convs.append(
                
                nn.Sequential(
                    nn.Conv1d(newInChannels,self.out_channels,self.k_size,self.stride_size,0,1,device=self.device),
                    nn.BatchNorm1d(self.out_channels,device=self.device),
                    nn.ReLU()
                    

                )


            )

        #the input size is the convolution reduced size, calculated by the number of     
        self.blstm = nn.LSTM(self.out_channels,self.hidden_size,2,batch_first=True,bidirectional=True) 
        
        

        #initialize style encoder components to run on content to encode (X2 or lower Branch)
        
        
    def forward(self,concat):

        """
        Performs encoder foward pass
        Parameters: concat: concatenated input of vocal spectrogram and speech style embedding duplicated T times 
        Returns: output: resulting encoded vocals
        """
        


        """
        newDimCopy = style[:,:,None]
        dup = torch.repeat_interleave(newDimCopy,spec.shape[2],2) #duplicate style embedding to be along with every single timestep

        concat = torch.cat((spec,dup))
        """
        output = concat
        for conv in self.convs:
            #print(f'new output shape is : {output.shape}')
            output = conv(output)
        

        #print(f'final output of convolution shape: {output.shape}')
        output = torch.transpose(output,2,1)

        output, cell = self.blstm(output) #do not use cell in this scenario

        fwd , back = torch.split(output,output.shape[2]//2,dim=2) #split into forward and backward components of BLSTM

        
        #print(f'fwd shape: {fwd.shape} back shape: {back.shape}')
        fwd = torch.nn.functional.interpolate(fwd,None,(self.downsamplefactor),'nearest') #downsample by desired downsampling factor 
        back = torch.nn.functional.interpolate(back,None,(self.downsamplefactor),'nearest')


        
        




        return fwd,back
    

        
