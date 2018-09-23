#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The 0.1 version of the training network for Tables-Embedding Project"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, COLUMN_DATA_TYPES=7, WORDLIST_LABEL_SIZE=4356):
        super(Net, self).__init__()
        # COLUMN_DATA_TYPES*10 input, 2 internal layer, 1*10 output
        self.COLUMN_DATA_TYPES = COLUMN_DATA_TYPES
        self.WORDLIST_LABEL_SIZE = WORDLIST_LABEL_SIZE
        self.fc1 = nn.Linear(self.COLUMN_DATA_TYPES * 10, 2 * self.COLUMN_DATA_TYPES * 10 + 10, True)
        self.fc2 = nn.Linear(2 * self.COLUMN_DATA_TYPES * 10 + 10, self.WORDLIST_LABEL_SIZE * 10, True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.normalize(x)
        return x

    def normalize(self, x):
        x.view(-1, self.WORDLIST_LABEL_SIZE)
        m = nn.Softmax()
        x = m(x).view(-1)
        return x

    def word_size(self):
        return self.WORDLIST_LABEL_SIZE

    def column_types(self):
        return self.COLUMN_DATA_TYPES


def save_model(model, name):
    torch.save(model.state_dict(), name)


def load_model(model, name):
    model.load_state_dict(torch.load(name))


def predict(net, input, batch_size=1):
    if batch_size > 1:
        output_ = []
        for i in range(batch_size):
            values, indices = net(torch.from_numpy(input[i]).float().view(-1).to(device)).view(-1, net.WORDLIST_LABEL_SIZE()).max(1)
            output = torch.zeros([net.WORDLIST_LABEL_SIZE(), 10])
            j = 0
            for indice in indices:
                output[indice][j] = 1
                j = j + 1
            output_.append(output)
        return output_

    elif batch_size == 1:
        output = net(torch.from_numpy(input).float().view(-1).to(device))
        values, indices = output.view(-1,net.WORDLIST_LABEL_SIZE()).max(1)
        output = np.zeros([net.WORDLIST_LABEL_SIZE(), 10])
        j = 0
        for indice in indices:
            output[indice][j] = 1
            j = j + 1
        return output
    else:
        print('error: batchsize no less than 1')
        exit(0)