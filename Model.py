from Encoder import Encoder
from PostNet import PostNet
from Vocoder import Vocoder
from Decoder import Decoder
import resemblyzer
import torch
import torch.nn as nn
import utils 



class Model(nn.Module):
    def __init__(self):
        #consider adding all component parameters here but alternatively we can assign them at each component instantiation
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.vocoder = Vocoder()
        self.postnet = PostNet()





    def forward(self,input):
        """
        forward pass of our model
        input: tuple of form (singing spec, speech style)
        
        """

        #depending on how we structure input, we will need to treat it slightly differently to start.
        #assume for now that spec comes first and style comes second in a tuple of inputs passed to this function

        spec = input[0]

        style= input[1]

        encoder_in = utils.combine_spec_and_style(spec,style)

        forward, backward = self.encoder.forward(encoder_in)


        #probably do loss calc here

        f_up = torch.nn.functional.interpolate(forward,None,32,'nearest') #upsampled forward output
        b_up = torch.nn.functional.interpolate(backward,None,32,'nearest') #upsampled backward output

        style_dup = torch.repeat_interleave(style,32,2) #duplicate style embedding desired number of times

        decoder_in = torch.cat((f_up,b_up,style_dup)) #do concatenation step of model, combining original style encoding with the output of the encoder

        decoder_out = self.decoder.forward(decoder_in)

        postnet_out = self.postnet.forward(decoder_out)

        out = self.vocoder(postnet_out)



        
        


