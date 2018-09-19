#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The 0.1 version of the training network for Tables-Embedding Project"""

import torch.save
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .__main__ import batch_size

COLUMN_DATA_TYPES = 7  # NUMBER OF POSSIBLE TYPES OF DATA IN A COLUMN OF A TABLE
WORDLIST_LABEL_SIZE = 300  # NUMBER OF DIFFERENT LABELS FOR WHOLE TRAINING SET


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # COLUMN_DATA_TYPES*10 input, 2 internal layer, 1*10 output
        self.fc1 = nn.Linear(COLUMN_DATA_TYPES * 10, 2 * WORDLIST_LABEL_SIZE * 10)
        self.fc2 = nn.Linear(2 * WORDLIST_LABEL_SIZE * 10, WORDLIST_LABEL_SIZE * 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.normalize(self, x)
        return x

    def normalize(self, x):
        x.view(-1, WORDLIST_LABEL_SIZE)
        xn = F.normalize(x, p=2, dim=1)
        xn = xn.view(-1)
        return xn

def save_model(model):
    model.save_state_dict("mytraining.pt")

def load_model(model):
    model.load_state_dict(torch.load("mytraining.pt"))

def predict(net, input, ifbatch = False):
    if ifbatch:
        output_ = []
        for i in range(batch_size):
            values, indices = torch.max(net(input), 0)
            output = np.zeros([10, WORDLIST_LABEL_SIZE])
            i = 0
            for indice in indices:
                output[0][indice] = 1
                i = i +1
            output_.append(output)
        return output_

    else:
        output = net(input)
        return output