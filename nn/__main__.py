#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plot
import train
import neural
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os

train_size = 50
batch_size = 50

def main():  #to be implemented
    batch_index = 0
    input, target = train.load_data(batch_size = batch_size, batch_index = batch_index)
    net = neural.Net(WORDLIST_LABEL_SIZE = np.shape(target[0][0]))
    if os.path.isfile('mytraining.pt'):
        print("=> loading checkpoint mytraining.pt")
        train.load_model(net,'mytraining.pt')
        print("=> loaded checkpoint mytraining.pt")
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    criterion = nn.MSELoss

    ###train part###
    while batch_size*batch_index<train_size:
        input, target = train.load_data(batch_size=batch_size, batch_index=batch_index)
        for i in range(batch_size):
            optimizer.zero_grad()
            output = net(input)
            target = target.view(-1, net.WORDLIST_LABEL_SIZE())
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        batch_index += 1
    train.save_model(net, 'mytraining.pt')
    ###train part end###

    ###test part###


    ###test part end###


if __name__ == '__main__':
    main()
