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
train_size = 10000
batch_size = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():  # to be implemented

    input, target = train.load_data(batch_size = 1, batch_index=0)

    file = open('log.txt','w')

    file.write('first load data')

    file.write('wordlist length = {}'.format(len(target[0])))

    net = neural.Net(WORDLIST_LABEL_SIZE = len(target[0])).to(device)

    print("nn prepared")

    if os.path.isfile('mytraining.pt'):
        file.write("=> loading checkpoint mytraining.pt")
        print("=> loading checkpoint mytraining.pt")
        neural.load_model(net, 'mytraining.pt')
        file.write("=> loaded checkpoint mytraining.pt")
        print("=> loaded checkpoint mytraining.pt")


    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    ###train part###

    train_accuracy = []
    train_accuracy_no_other = []
    validation_accuracy = []
    validation_accuracy_no_other = []
    iteration = 0
    '''
    while(iteration < 10):

        iteration += 1

        for ite in range(100):

            batch_index = 0

            print("start training iteration {} // {} \n".format(iteration, ite))
            file.write("start training iteration {} // {} \n".format(iteration, ite))

            while batch_size * batch_index < train_size:
                input, target = train.load_data(batch_size=batch_size, batch_index=batch_index)

                for i in range(batch_size):

                    target_ = torch.from_numpy(target[i]).float().view(-1).to(device)  # fix for critetion
                    output = net(torch.from_numpy(input[i]).float().view(-1).to(device))

                    loss = criterion(output, target_)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_index += 1

            neural.save_model(net, 'mytraining.pt')
            print('model saved')
            file.write('model saved iteration{} : ite{}\n'.format(iteration, ite))

        print("end training iteration {}\n".format(iteration))
        file.write("end training iteration {}\n".format(iteration))
        '''


    with torch.no_grad():

        file.write("start predict iteration {}\n".format(iteration))
        print("start predict train iteration {}\n".format(iteration))
        accuracy = []
        accuracy_no_other = []
        for test_index in range(int(round(train_size/batch_size))):
            file.write('train accuracy batch index {}\n'.format(test_index))
            print('train accuracy batch index {}\n'.format(test_index))
            input, target = train.load_data(batch_size=batch_size, batch_index=test_index)
            target = torch.from_numpy(target).float()
            prediction = neural.predict(net, input, batch_size)
            prediction_no_other = neural.predict(net, input, batch_size)
            accuracy.append(train.accuracy(prediction, target, batch_size))
            accuracy_no_other.append(train.accuracy_no_other(prediction_no_other, target, batch_size))
            file.write('train accuracy batch index {} end\n'.format(test_index))
            print('train accuracy batch index {} end\n'.format(test_index))
        train_accuracy.append(np.average(np.average(np.array(accuracy))))
        train_accuracy_no_other.append(np.average(np.average(np.array(accuracy_no_other))))
        file.write('iteration {} train_accuracy {}\n'.format(iteration, train_accuracy))
        file.write('train_accuracy_no_other {}\n'.format(train_accuracy_no_other))
        print(train_accuracy)
        print(train_accuracy_no_other)

    with torch.no_grad():

        print("start predict validation iteration {}\n".format(iteration))
        accuracy = []
        accuracy_no_other = []
        for test_index in range(int(round(train_size/batch_size)),int(round((train_size+1000)/batch_size))):
            input, target = train.load_data(batch_size=batch_size, batch_index=test_index)
            target = torch.from_numpy(target).float()
            prediction = neural.predict(net, input, batch_size)
            prediction_no_other = neural.predict(net, input, batch_size)
            accuracy.append(train.accuracy(prediction, target, batch_size))
            accuracy_no_other.append(train.accuracy_no_other(prediction_no_other, target, batch_size))
        print(accuracy)
        print(accuracy_no_other)
        validation_accuracy.append(np.average(np.average(np.array(accuracy))))
        validation_accuracy_no_other.append(np.average(np.average(np.array(accuracy_no_other))))
        file.write('iteration {} validation_accuracy {}\n'.format(iteration, validation_accuracy))
        file.write('validation_accuracy_no_other {}\n'.format(validation_accuracy_no_other))
        print(validation_accuracy)
        print(validation_accuracy_no_other)



    # print('ploting accuracy graph')
    # plot.plot_accuracy_over_iteration(train_accuracy, validation_accuracy, iteration, True)
    # plot.plot_accuracy_no_other_over_iteration(train_accuracy_no_other, validation_accuracy_no_other, True)

    print("end training")



    ###train part end###

    ###test part###



    ###test part end###

    #print(train.accuracy_no_other(torch.from_numpy(np.array([[0,1,0],[1,0,1]])),torch.from_numpy(np.array([[0,1,0],[0, 0, 1]]))))

if __name__ == '__main__':
    main()
