from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import glob
import os
import pylidc as pl

class Dataset(BaseDataset):
    def __init__(self,):
        
        self.data = [
                {
                    # img: 普通の画像,
                    # mask: マスク画像
                    }
                ]

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
