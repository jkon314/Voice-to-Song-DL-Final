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

        # print(f'spec shape: {spec.shape} style shape: {style.shape}')

        encoder_in = utils.combine_spec_and_style(spec,style)
        # print("Encoder In: ", encoder_in.shape)
        
        forward, backward = self.encoder.forward(encoder_in)
        # print("Encoder PASSED!")

        #probably do loss calc here

        f_up = torch.nn.functional.interpolate(forward,None,32,'nearest') #upsampled forward output
        b_up = torch.nn.functional.interpolate(backward,None,32,'nearest') #upsampled backward output

        # print(f'forward shape: {forward.shape} backward shape: {backward.shape}')
        encoder_out = torch.cat((forward,backward),2)

        # print("Encoder Out: ", encoder_out.shape)


        style = style[:,:,None]
        style_dup = torch.repeat_interleave(style,f_up.shape[1],2) #duplicate style embedding desired number of timesteps
        style_dup = torch.transpose(style_dup,2,1) #change shape to match upsampled encoder outputs




        


        decoder_in = torch.cat((f_up,b_up,style_dup),2) #do concatenation step of model, combining original style encoding with the output of the encoder

        decoder_in = torch.transpose(decoder_in,1,2)

        # print("Decoder In: ", decoder_in.shape)

        

        decoder_out = self.decoder.forward(decoder_in)
        # print("Decoder Out: ", decoder_out.shape)
        
        postnet_out = self.postnet.forward(decoder_out) + decoder_out
        # print("Postnet Out: ", postnet_out.shape)

        # print("Style: ", style.shape)
        encoded_postnet_in = utils.combine_spec_and_style(postnet_out,style.squeeze())
        # print("Encoded Postnet In: ", encoded_postnet_in.shape)



        encoded_postnet_out_f, encoded_postnet_out_b = self.encoder.forward(encoded_postnet_in)
        encoded_postnet_out = torch.cat((encoded_postnet_out_f,encoded_postnet_out_b),2)
        # print("Encoded Postnet Out: ", encoded_postnet_out.shape)

        # encoded_postnet_out = None


        # print("MODEL PASSED!")
        return spec, encoder_out, decoder_out, postnet_out, encoded_postnet_out



        
        


