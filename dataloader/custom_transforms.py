import torch
import numpy as np
from PIL import Image

"""
This module is custom transforms of image data.
This is called in <dataloader.dataset.Dataset> as tr.
You can add your custom transforms in this module and call it in <dataset>.
REMIND each input and output should be numpy ndarray (except ToTensor())
to be modulalization.
[Pre-implemented]
Normalize:
ToTensor:
Resize:
"""


class Resize(object):
    """
    Reshape a tensor image with size.
    """
    def __init__(self, size):
        self.size_w = size[0]
        self.size_h = size[1]
    
    def __call__(self, sample):
        img = sample["input"]
        target = sample["label"]
        
        img = Image.fromarray(np.uint8(img))
        img = np.asarray(img.resize((self.size_w, self.size_h)))
        
        return {"input": img,
                "label": target}
    

class Normalize(object):
    """
    Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=-580.0195, std=453.7174):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample["input"]
        target = sample["label"]
        
        img = img.astype(np.float32)
        img -= self.mean
        img /= self.std

        return {"input": img,
                "label": target}
    

class PaddingSurround(object):
    def __call__(self, sample):
        img = sample["input"]
        target = sample["label"]
        global_min = img.min()
        local_min = img[img>global_min].min()
        img[img == global_min] = local_min

        return {"input": img,
                "label": target}

class UpperThreshold(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, sample):
        img = sample["input"]
        target = sample["label"]

        img[img>self.threshold] = self.threshold

        return {"input": img,
                "label": target}

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample["input"])
        target = np.array(sample["label"])
        
        img = img.reshape(img.shape + (1, ))
        rgb_img = np.append(img, img, axis=2)
        rgb_img = np.append(rgb_img, img, axis=2)
        # print (rgb_img.shape)
        img = rgb_img.astype(np.float32).transpose((2, 0, 1))
        
        img = torch.from_numpy(img).float()
        target = torch.from_numpy(target).long()

        return {"input": img,
                "label": target}
