import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

class Modeling(nn.Module):
    def __init__():
        super(Modeling, self).__init__()
        self.unet = smp.Unet('resneet34', encoder_weight='imagenet')

    def forward(self, x):
        x = self.unet(x)
        return x
