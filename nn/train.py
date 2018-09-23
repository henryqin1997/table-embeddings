import os
import json
import numpy
import torch.nn as nn
import torch.optim as optim
from .__main__ import batch_size

from neural import Net, WORDLIST_LABEL_SIZE

training_data_dir = './data/train'
training_files_json = './data/training_files.json'
training_files = json.load(open(training_files_json))


def load_data(batch_size, batch_index=0):
    # load training data from file, to be implemented
    # put size number of data into one array
    # start from batch_index batch
    batch_files = training_files[batch_size * batch_index:batch_size * (batch_index + 1)]
    batch_files_ner = list(map(lambda batch_file: batch_file.rstrip('.json') + '_ner.csv', batch_files))
    batch_files_wordlist = list(map(lambda batch_file: batch_file.rstrip('.json') + '_wordlist.csv', batch_files))
    input = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_ner), delimiter=',') for batch_file_ner in
         batch_files_ner])
    target = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_wordlist), delimiter=',') for batch_file_wordlist
         in batch_files_wordlist])
    return input, target


##########################333#3#
# evaluation of model:
# 1. accuracy of prediction of label over target     #correct prediction/#targetlabel
# 2. accuracy of prediction of label over target when other doesn't count
#   #correct prediction(no 'other')/#targetlabel(no other)


def accuracy(prediction,target,batch_size):  # to be implemented
    total_num=0
    correct_num=0
    batch_size = target.shape[0]
    label_size = target.shape[1]
    col_size = target.shape[2]
    for batch_index in range(batch_size):
        for col_index in range(col_size):
            for label_index in range(label_size):
                if target[batch_index,label_index,col_index]==1:
                    if prediction[batch_index,label_index,col_index]==1 and target[batch_index,label_index,col_index]==1:
                        correct_num=correct_num+1
            total_num =total_num+numpy.sum(prediction[batch_index,:,col_index])
    return correct_num/total_num


def accuracy_no_other(prediction,target):  # to be implemented
    total_num=0
    correct_num=0
    batch_size=target.shape[0]
    label_size=target.shape[1]-1
    col_size=target.shape[2]
    for batch_index in range(batch_size):
        for col_index in range(col_size):
            for label_index in range(label_size):
                if target[batch_index,label_index,col_index]==1:
                    if prediction[batch_index,label_index,col_index]==1 and target[batch_index,label_index,col_index]==1:
                        correct_num=correct_num+1
            total_num =total_num+numpy.sum(prediction[batch_index,:,col_index])
    return correct_num/total_num


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
