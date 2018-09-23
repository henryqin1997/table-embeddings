#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import neural
import torch
import torch.nn as nn
import train
import plot

train_size = 2000
batch_size = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():  # to be implemented
    batch_index = 0
    input, target = train.load_data(batch_size = 1, batch_index=batch_index)
    print('first load data')

    print('wordlist length = {}'.format(len(target[0])))

    net = neural.Net(WORDLIST_LABEL_SIZE = len(target[0])).to(device)

    print("nn prepared")

    if os.path.isfile('mytraining.pt'):
        print("=> loading checkpoint mytraining.pt")
        train.load_model(net, 'mytraining.pt')
        print("=> loaded checkpoint mytraining.pt")


    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    ###train part###
    '''
    while batch_size * batch_index < train_size:

        input, target = train.load_data(batch_size=batch_size, batch_index=batch_index)
        print("start training")
        for i in range(batch_size):
            print(batch_size*batch_index+i)

            target_ = torch.from_numpy(target[i]).float().view(-1).to(device)  # fix for critetion
            output = net(torch.from_numpy(input[i]).float().view(-1).to(device))

            loss = criterion(output, target_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_index += 1

    print("end training")
    
    neural.save_model(net, 'mytraining.pt')
    print('model saved')
    ###train part end###
    '''
    ###test part###

    with torch.no_grad():
        print("start predict")
        accuracy = []
        accuracy_no_other = []
        for test_index in range(39, 40):
            input, target = train.load_data(batch_size=batch_size, batch_index=test_index)
            prediction = neural.predict(net, input, ifbatch=True)
            prediction_no_other = neural.predict(net, input, ifbatch=True)
            accuracy.append(train.accuracy(prediction, target, ifbatch = True))
            accuracy_no_other.append(train.accuracy_no_other(prediction_no_other, target, ifbatch = True))
        plot.plottvsv(accuracy,accuracy_no_other,batch_size) #this is only for test
        print(accuracy)
        print(accuracy_no_other)

    ###test part end###


if __name__ == '__main__':
    main()
