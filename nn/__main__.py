#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .neural import Net, load_model, save_model
from .train import train, load_data, accuracy
import os

train_size = 10000
batch_size = 100

def main():  #to be implemented
    net = Net()
    if os.path.isfile('mytraining.pt'):
        print("=> loading checkpoint mytraining.pt")
        load_model(net)
        print("=> loaded checkpoint mytraining.pt")





    input, target = load_data()




if __name__ == '__main__':
    main()
