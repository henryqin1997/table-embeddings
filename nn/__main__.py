#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import neural
import numpy as np
import torch
import torch.nn as nn
import train
import plot

iteration_size = 200
train_size = 1950
batch_size = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():  # to be implemented
    input, target = train.load_data(batch_size = 1, batch_index=0)

    file = open('log.txt','w')

    file.write('first load data')

    file.write('wordlist length = {}'.format(len(target[0])))

    net = neural.Net(WORDLIST_LABEL_SIZE = len(target[0])).to(device)

    file.write("nn prepared")

    if os.path.isfile('mytraining.pt'):
        file.write("=> loading checkpoint mytraining.pt")
        neural.load_model(net, 'mytraining.pt')
        file.write("=> loaded checkpoint mytraining.pt")


    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    ###train part###
    train_accuracy = []
    train_accuracy_no_other = []
    validation_accuracy = []
    validation_accuracy_no_other = []
    iteration = 0

    while(iteration < 20):

        batch_index = 0

        iteration += 1

        with torch.no_grad():

            file.write("start predict iteration {}".format(iteration))
            print("start predict iteration {}".format(iteration))
            accuracy = []
            accuracy_no_other = []
            for test_index in range(2):
                file.write('train accuracy batch index {}'.format(test_index))
                print('train accuracy batch index {}'.format(test_index))
                input, target = train.load_data(batch_size=batch_size, batch_index=test_index)
                target = torch.from_numpy(target).float()
                prediction = neural.predict(net, input, batch_size)
                prediction_no_other = neural.predict(net, input, batch_size)
                accuracy.append(train.accuracy(prediction, target, batch_size))
                accuracy_no_other.append(train.accuracy_no_other(prediction_no_other, target, batch_size))
                file.write('train accuracy batch index {} end'.format(test_index))
                print('train accuracy batch index {} end'.format(test_index))
            file.write('{}'.format(accuracy))
            file.write('{}'.format(accuracy_no_other))
            train_accuracy.append(np.average(np.average(np.array(accuracy))))
            train_accuracy_no_other.append(np.average(np.average(np.array(accuracy_no_other))))
            file.write('{}'.format(train_accuracy))
            file.write('{}'.format(train_accuracy_no_other))
            return 0

        with torch.no_grad():

            print("start predict iteration {}".format(iteration))
            accuracy = []
            accuracy_no_other = []
            for test_index in range(39, 40):
                input, target = train.load_data(batch_size=27, batch_index=test_index)
                target = torch.from_numpy(target).float()
                prediction = neural.predict(net, input, 27)
                prediction_no_other = neural.predict(net, input, 27)
                accuracy.append(train.accuracy(prediction, target, 27))
                accuracy_no_other.append(train.accuracy_no_other(prediction_no_other, target, 27))
            plot.plottvsv(accuracy,accuracy_no_other,batch_size) #this is only for test
            print(accuracy)
            print(accuracy_no_other)
            validation_accuracy.append(np.average(np.array(accuracy)))
            validation_accuracy_no_other.append(np.average(np.array(accuracy_no_other)))

        while batch_size * batch_index < train_size:
            input, target = train.load_data(batch_size=batch_size, batch_index=batch_index)
            print("start training iteration {}".format(iteration))
            for i in range(batch_size):
                print(batch_size*batch_index+i)

                target_ = torch.from_numpy(target[i]).float().view(-1).to(device)  # fix for critetion
                output = net(torch.from_numpy(input[i]).float().view(-1).to(device))

                loss = criterion(output, target_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_index += 1

    print('ploting accuracy graph')
    plot.plot_accuracy_over_iteration(train_accuracy, validation_accuracy, iteration, True)
    plot.plot_accuracy_no_other_over_iteration(train_accuracy_no_other, validation_accuracy_no_other, True)



    print("end training")
    
    neural.save_model(net, 'mytraining.pt')
    print('model saved')
    ###train part end###

    ###test part###



    ###test part end###


if __name__ == '__main__':
    main()
