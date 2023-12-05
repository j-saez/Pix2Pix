import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, downsample: bool, drop: bool, add_norm: bool, act_function: str, stride: int=2) -> None:
        """
        Convolutional Block.
        Inputs:
            >> in_ch:          (int) Number of input channels
            >> out_ch:         (int) Number of output channels
            >> downsample:     (bool) Downsample (True) or upsample (False).
            >> drop:           (bool) Add dropout (True) or not (False).
            >> add_norm:       (bool) Add instance normalization layer.
            >> act_function:   (str) relu or lrelu

        Attributes:
            >> conv:           (nn.Module) Convolutional or TransposeConv layer.
            >> bn:             (nn.Module) Instance normalization layer.
            >> activation:     (nn.ReLU or nn.LeakyReLU)
            >> dropuout:       (nn.Module) Dropout layer.
            >> drop:           (bool) Wheter or not to apply dropoout
            >> add_norm:       (bool) Whether to apply or not instance normalization.
        """
        super().__init__()

        if act_function != 'relu' and act_function != 'lrelu':
            raise ValueError(f'act_function must be relu or lrelu. Received: {act_function}')

        # If using instance norm True, if using batch norm false
        use_bias = True
        downsample_conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=use_bias)
        upsample_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=use_bias)

        # Layer
        self.conv = downsample_conv if downsample else upsample_conv
        self.bn = nn.InstanceNorm2d(out_ch)
        self.activation = nn.ReLU() if act_function == 'relu' else nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)

        self.drop = drop
        self.add_norm = add_norm
        return

    def forward(self, inputs: torch.Tensor):
        x = self.conv(inputs)
        if self.add_norm: x = self.bn(x)
        if self.drop: x = self.dropout(x)
        x = self.activation(x)
        return x

