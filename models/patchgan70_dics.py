import torch
import torch.nn as nn
from models.ck import ConvBlock

"""
C64-C128-C256-C512
"""

class PatchGAN70Disc(nn.Module):

    def __init__(self, in_chs) -> None:
        """
        PatchGAN70Disc
        Inputs:
            >> in_chs: (int) Number of input channels.
        """
        super().__init__()

        self.model = nn.Sequential(
            ConvBlock(in_chs*2, 64, downsample=True, drop=False, add_norm=False, act_function='lrelu'),
            ConvBlock(64,  128, downsample=True, drop=False, add_norm=True, act_function='lrelu'),
            ConvBlock(128, 256, downsample=True, drop=False, add_norm=True, act_function='lrelu'),
            ConvBlock(256, 512, downsample=True, drop=False, add_norm=True, act_function='lrelu', stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True, padding_mode="zeros")
        )
        

    def forward(self, source_dom_imgs: torch.Tensor, target_dom_imgs: torch.Tensor):
        """
        inputs: (torch.Tensor [batch, chs, h, w])
        """
        inputs = torch.cat((source_dom_imgs, target_dom_imgs), dim=1)
        outputs = self.model(inputs)
        return outputs
