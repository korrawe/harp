""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, n_latent=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.n_latent = n_latent

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor + self.n_latent)
        self.up1 = Up(1024 + self.n_latent * factor, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, latent=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # import pdb; pdb.set_trace()
        if latent is not None:
            latent = latent.unsqueeze(-1).unsqueeze(-1)
            latent = latent.repeat(1, 1, x5.shape[2], x5.shape[3])
            x5 = torch.cat([x5, latent], dim=1)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.sigmoid(logits)