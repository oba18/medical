import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Modules
from utils.loss import SegmentationLosses 
from dataloader.dataset import Dataset
from modeling.model import Modeling

class Trainer(object):
    def __init__(
            self,
            batch_size=12,
            optimizer_name='Adam',
            lr=1e-3,
            weight_decay=1e-5,
            epochs=200,
            mode
            )
