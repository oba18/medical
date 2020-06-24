import os
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Project Modules
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.optimizer import Optimizer
from utils.loss import SegmentationLosses, BCEDiceLoss
from dataloader.dataset import Dataset
from dataloader import *
from modeling.model import Modeling
from my_setting import pycolor

def predict():

    batch_size = 32
    is_develop = True

    # load data
    train_loader, val_loader, test_loader, num_classes = make_data_loader(batch_size, is_develop=is_develop)

    # load model
    model = Modeling(num_classes)
    
    # load evaluator
    evaluator = Evaluator(num_classes)

    # load weight
    path = './run/model02/model_best.pth.tar'
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    for i, sample in enumerate(test_loader):
        inputs, target = sample["input"], sample["label"]
        output = model(inputs)
        output = torch.argmax(output, axis=1).data.cpu().numpy()
        output = output.transpose((1, 2, 0))
        inputs = inputs.data.cpu().numpy()
        
        # outputを8ビットに変換する
        mean = -580.0195
        std = 453.7174
        output = abs((output * std + mean) * 255 / 1000)

        # 画像保存
        cv2.imwrite(f'/output_images/{i}.jpg', output)
    return output.dtype


print (predict())
# predict()
