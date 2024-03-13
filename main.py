import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
#from typing import Sequence
from torchvision.transforms import functional as F
#import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
#import torchmetrics as TM
from dataclasses import dataclass
import dataclasses

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

# Set the working (writable) directory.
working_dir = "/content/"

from utils.model import get_fcn_model, get_unet_model
from utils.helpers import *
from utils.dataset import *
from utils.validation import IoULoss
#from utils.test import *
from utils.train_epoch import train_model

mode = 'train'
dataset_path = '/content/train_set/'
#mode = 'test'
#dataset_path = '/content/test_set/'
image_lst, mask_lst = prepareList(dataset_path, mode)

model = get_fcn_model(pretrained = True, orig = False)

to_device(model)
my_dataset = SegmentationDataSet(image_lst, mask_lst)
train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=16, shuffle=True)
params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
optimizer = torch.optim.Adam(params, lr = 0.0004)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.8)

def train_loop(model, loader, epochs, optimizer, scheduler, save_path):
    epoch_i, epoch_j = epochs
    for i in range(epoch_i, epoch_j):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        train_model(model, loader, optimizer)
        if scheduler is not None:
            scheduler.step()
        print("")

if __name__ == '__main__':
  train_loop(model, train_loader, (1, 20), optimizer, scheduler, save_path=None)
