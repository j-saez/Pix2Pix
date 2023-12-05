import torch
import torch.nn as nn
from models.ck import ConvBlock

"""
encoder: C64-C128-C256-C512-C512-C512-C512-C512
decoder: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
"""

class UNETGenerator(nn.Module):

    def __init__(self, in_chs, out_chs) -> None:
        """
        UNET generator class
        Inputs:
            >> in_chs: (int) Number of input channels.
            >> out_chs: (int) Number of output channels.
        """
        super().__init__()
        
        # Encoder
        ## Following the paper, batch norm not applied to the first layer
        self.in_l = ConvBlock(in_chs, 64, downsample=True, drop=False, add_norm=False, act_function='lrelu')
        self.enc_l1 = ConvBlock(64,  128, downsample=True, drop=False, add_norm=True, act_function='lrelu') # 64 x 64
        self.enc_l2 = ConvBlock(128, 256, downsample=True, drop=False, add_norm=True, act_function='lrelu') # 32 x 32
        self.enc_l3 = ConvBlock(256, 512, downsample=True, drop=False, add_norm=True, act_function='lrelu') # 16 x 16
        self.enc_l4 = ConvBlock(512, 512, downsample=True, drop=False, add_norm=True, act_function='lrelu') # 8 x 8
        self.enc_l5 = ConvBlock(512, 512, downsample=True, drop=False, add_norm=True, act_function='lrelu') # 4 x 4
        self.enc_l6 = ConvBlock(512, 512, downsample=True, drop=False, add_norm=True, act_function='lrelu') # 2 x 2
        self.enc_l7 = ConvBlock(512, 512, downsample=True, drop=False, add_norm=False, act_function='lrelu') # 1 x 1

        # Decoder
        self.dec_l1 = ConvBlock(512,   512, downsample=False, drop=True, add_norm=True, act_function='relu') # 2 x 2
        self.dec_l2 = ConvBlock(512*2, 512, downsample=False, drop=True, add_norm=True, act_function='relu') # 4 x 4
        self.dec_l3 = ConvBlock(512*2, 512, downsample=False, drop=True, add_norm=True, act_function='relu') # 8 x 8
        self.dec_l4 = ConvBlock(512*2, 512, downsample=False, drop=False, add_norm=True, act_function='relu') # 16 x 16
        self.dec_l5 = ConvBlock(512*2, 256, downsample=False, drop=False, add_norm=True, act_function='relu') # 32 x 32
        self.dec_l6 = ConvBlock(256*2, 128, downsample=False, drop=False, add_norm=True, act_function='relu') # 64 x 64
        self.dec_l7 = ConvBlock(128*2, 64,  downsample=False, drop=False, add_norm=True, act_function='relu') # 128 x 128
        self.dec_ouptput_layer = nn.Sequential( # 256 x 256
            nn.ConvTranspose2d(64*2, out_chs, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),
        )
        return

    def forward(self, inputs: torch.Tensor):
        """
        inputs: (torch.Tensor [batch, chs, h, w])
        """

        # Encode
        in_l_out = self.in_l(inputs)
        enc_l1_out = self.enc_l1(in_l_out)
        enc_l2_out = self.enc_l2(enc_l1_out)
        enc_l3_out = self.enc_l3(enc_l2_out)
        enc_l4_out = self.enc_l4(enc_l3_out)
        enc_l5_out = self.enc_l5(enc_l4_out)
        enc_l6_out = self.enc_l6(enc_l5_out)
        enc_l7_out = self.enc_l7(enc_l6_out)

        # Decode
        dec_l1_out = self.dec_l1(enc_l7_out)
        dec_l2_out = self.dec_l2(torch.cat((dec_l1_out,enc_l6_out),dim=1))
        dec_l3_out = self.dec_l3(torch.cat((dec_l2_out,enc_l5_out),dim=1))
        dec_l4_out = self.dec_l4(torch.cat((dec_l3_out,enc_l4_out),dim=1))
        dec_l5_out = self.dec_l5(torch.cat((dec_l4_out,enc_l3_out),dim=1))
        dec_l6_out = self.dec_l6(torch.cat((dec_l5_out,enc_l2_out),dim=1))
        dec_l7_out = self.dec_l7(torch.cat((dec_l6_out,enc_l1_out),dim=1))
        output = self.dec_ouptput_layer(torch.cat((dec_l7_out, in_l_out),dim=1))
        
        
        return output

