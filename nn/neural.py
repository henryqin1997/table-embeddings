#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The 0.1 version of the training network for Tables-Embedding Project"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import __main__


class Net(nn.Module):
    def __init__(self, COLUMN_DATA_TYPES=7, WORDLIST_LABEL_SIZE=4356):
        super(Net, self).__init__()
        # COLUMN_DATA_TYPES*10 input, 2 internal layer, 1*10 output
        self.COLUMN_DATA_TYPES = COLUMN_DATA_TYPES
        self.WORDLIST_LABEL_SIZE = WORDLIST_LABEL_SIZE
        self.fc1 = nn.Linear(self.COLUMN_DATA_TYPES * 10, 2 * self.WORDLIST_LABEL_SIZE * 10, True)
        self.fc2 = nn.Linear(2 * self.WORDLIST_LABEL_SIZE * 10, self.WORDLIST_LABEL_SIZE * 10, True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.normalize(x)
        return x

    def normalize(self, x):
        x.view(-1, self.WORDLIST_LABEL_SIZE)
        xn = F.normalize(x, p=2, dim=1)
        xn = xn.view(-1)
        return xn

    def word_size(self):
        return self.WORDLIST_LABEL_SIZE

    def column_types(self):
        return self.COLUMN_DATA_TYPES


def save_model(model, name):
    model.save_state_dict(name)


def load_model(model, name):
    model.load_state_dict(torch.load(name))


def predict(net, input, ifbatch=False):
    if ifbatch:
        output_ = []
        for i in range(__main__.batch_size):
            values, indices = net(input[i]).view(-1,net.WORDLIST_LABEL_SIZE()).max(1)
            output = np.zeros([10, net.WORDLIST_LABEL_SIZE()])
            j = 0
            for indice in indices:
                output[j][indice] = 1
                j = j + 1
            output_.append(output)
        return output_

    else:
        output = net(input)
        values, indices = output.view(-1,net.WORDLIST_LABEL_SIZE()).max(1)
        output = np.zeros([10, net.WORDLIST_LABEL_SIZE()])
        j = 0
        for indice in indices:
            output[j][indice] = 1
            j = j + 1
        return output
