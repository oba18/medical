import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

class Modeling(nn.Module):
    def __init__(self, num_class):
        super(Modeling, self).__init__()
        self.unet = smp.Unet('resnet34', encoder_weights='imagenet',  classes=num_class)

    def forward(self, x):
        x = self.unet(x)
        return x

class ModelingLookTwice(nn.Module):
    def __init__(self, num_class):
        super(Modeling, self).__init__()
        self.unet = smp.Unet('resnet34', encoder_weights='imagenet',  classes=num_class)

    def forward(self, x):
        # Shape of x is [B, C, W, H]
        x1 = torch.cat(x, torch.zeros((x.shape[0], 2, x.shape[2], x.shape[3])), 1)
        y1 = self.unet(x1)
        x2 = torch.cat(x, y1, 1)
        y2 = self.unet(x2)
        return y1, y2
