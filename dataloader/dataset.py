from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import glob
import os
import pylidc as pl
import pydicom
import cv2
from sklearn.utils import shuffle
import dataloader.custom_transforms as tr
from my_setting import MyPath
from tqdm import tqdm 
from torchvision import transforms

class Dataset(BaseDataset):
    NUM_CLASSES = MyPath.num_class
    def __init__(self, img_list, mask_list, split='train'):
        self.split = split
        img_list, mask_list = shuffle(img_list, mask_list, random_state = 1)
        train_len = int(len(img_list) * 0.8) # 訓練データ
        val_len = int(len(img_list) * 0.1) # 検証データ
        test_len = int(len(img_list) - train_len - val_len) # テストデータ
        if split == 'train':
            self.img_list, self.mask_list = img_list[:train_len], mask_list[:train_len]
        elif split == 'val':
            self.img_list, self.mask_list = img_list[train_len:train_len + val_len], mask_list[train_len:train_len + val_len]
        elif split == 'test':
            self.img_list, self.mask_list = img_list[train_len + val_len:], mask_list[train_len + val_len:]

    def __getitem__(self, index):
        sample =  {
                'input': self.img_list[index],
                'label': self.mask_list[index]
                }
        if self.split == 'train':
            return self.transform_tr(sample)
        if self.split == 'val':
            return self.transform_val(sample)
        if self.split == 'test':
            return self.transform_val(sample)
        return sample
    
    def __len__(self):
        return len(self.img_list)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.Resize(size=(64, 64)),
            tr.PaddingSurround(),
            tr.UpperThreshold(0),
            # tr.Normalize(mean=0, std=1),
            tr.ToTensor()
            ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # tr.Resize(size=(64, 64)),
            tr.Normalize(mean=0, std=1),
            tr.ToTensor()
            ])
        return composed_transforms(sample)


