import os

import torch.nn as nn
import torch.optim as optim

from neural import Net, WORDLIST_LABEL_SIZE


def load_data():
    # load training data from file, to be implemented
    input = 0
    target = 0
    return input, target

##########################333#3#
#evaluation of model:
#1. accuracy of prediction of label over target     #correct prediction/#targetlabel
#2. accuracy of prediction of label over target when other doesn't count
#   #correct prediction(no 'other')/#targetlabel(no other)


def accuracy(): # to be implemented
    return 0

def accuracy_no_other(): # to be implemented
    return 0

################################


def train(input, target, net):
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    optimizer.zero_grad()
    output = net(input)
    target = target.view(-1, WORDLIST_LABEL_SIZE)
    criterion = nn.MSELoss
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()



