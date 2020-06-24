import torch
import torch.nn as nn
import torch.nn.functional as F

# import segmentation_models_pytorch as smp
from .unet import *

class Modeling(nn.Module):
    def __init__(self, num_class):
        super(Modeling, self).__init__()
        # self.unet = smp.Unet('resnet34', encoder_weights='imagenet',  classes=num_class)
        self.unet = UNet(n_channels=3, n_classes=num_class)

    def forward(self, x):
        x = self.unet(x)
        return x
