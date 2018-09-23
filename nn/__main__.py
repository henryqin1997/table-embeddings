#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .neural import Net, load_model, save_model, predict
from .train import train, load_data, accuracy
import numpy as np
import os

train_size = 50
batch_size = 50

def main():  #to be implemented
    batch_index = 0
    input, target = load_data(batch_size = batch_size, batch_index = batch_index)
    net = Net(WORDLIST_LABEL_SIZE = np.shape(target[0][0]))
    if os.path.isfile('mytraining.pt'):
        print("=> loading checkpoint mytraining.pt")
        load_model(net,'mytraining.pt')
        print("=> loaded checkpoint mytraining.pt")
    while batch_size*batch_index<train_size:
        input, target = load_data(batch_size=batch_size, batch_index=batch_index)
        for i in range(batch_size):
            train(input, net, target)
    save_model(net, 'mytraining.pt')



if __name__ == '__main__':
    main()
