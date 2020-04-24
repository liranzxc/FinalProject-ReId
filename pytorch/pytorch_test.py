#!/usr/bin/env python
# coding: utf-8

# In[1]:
from builtins import next, iter

import torch
import torchreid
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


# In[2]:
from pair_dataset import PairDataset


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),  # 128@42*42
            nn.MaxPool2d(2),  # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),  # 128@18*18
            nn.MaxPool2d(2),  # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),  # 256@4*4
        )

        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(-1, 4 * 4 * 256)
        x = torch.sigmoid(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out


# for test
net = Siamese()


# In[3]:

# functions to show an image

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# In[4]:

datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=96,
    width=96,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)


db_new = PairDataset(datamanager)


