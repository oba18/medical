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
from PIL import Image

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
    num_classes = 2 

    # load data
    test_loader = make_data_loader(batch_size, is_develop=is_develop)[2]

    # load model
    model = Modeling(num_classes)
    
    # load evaluator
    evaluator = Evaluator(num_classes)

    # load weight
    path = './run/model02/model_best.pth.tar'
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    for sample in test_loader:
        inputs, target = sample["input"], sample["label"]
        output = model(inputs)
        print ('1', output.shape, type(output))

        ##### output #####
        output = torch.argmax(output, axis=1).data.cpu().numpy()
        #  output = output.transpose((1,2,0))
        #  output = output.data.cpu().numpy()
        #  output = output.transpose((0, 2, 3, 1))
        output = output.astype(np.uint8)
        #  print (output.shape, output.dtype)

        ###### input #####
        input = inputs.data.cpu().numpy()
        input = input.transpose((0, 2, 3, 1))

        mean = -580.0195
        std = 453.717

        input = (input * std + mean).astype(np.uint8)

        for n, (i, o) in enumerate(zip(input, output)):
            print (i.shape)
            print (o.shape)
            img = i[:,:,0] * o
            Image.fromarray(img).save(f'./output_images/{n}.png')


print (predict())
# predict()
