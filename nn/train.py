import json
import os

import numpy

training_data_dir = '../data/train'
training_files_json = '../data/training_files_filtered.json'
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

def accuracy(prediction, target, batch_size=1):  # to be implemented
    if batch_size>1:
        total_num = 0
        correct_num = 0
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                correct_num=correct_num+int(prediction[batch_index][:,col_index].dot(target[batch_index][:,col_index]))
                total_num = total_num + int(sum(prediction[batch_index][:, col_index]))
        return correct_num / total_num
    else:
        total_num = 0
        correct_num = 0
        col_size = target.shape[1]
        for col_index in range(col_size):
            correct_num = correct_num + int(prediction[:, col_index].dot(target[:, col_index]))
            total_num = total_num + int(sum(prediction[:, col_index]))
        return correct_num / total_num


def accuracy_no_other(prediction, target, batch_size=1):  # to be implemented
    if batch_size>1:
        total_num = 0
        correct_num = 0
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                correct_num=correct_num+int(prediction[batch_index][:-1,col_index].dot(target[batch_index][:-1,col_index]))
                total_num = total_num + int(sum(prediction[batch_index][:-1, col_index]))
        return correct_num / total_num
    else:
        total_num = 0
        correct_num = 0
        col_size = target.shape[1]
        for col_index in range(col_size):
            correct_num = correct_num + int(prediction[:-1, col_index].dot(target[:-1, col_index]))
            total_num = total_num + int(sum(prediction[:-1, col_index]))
        return correct_num / total_num

def accuracy_possibility(prediction_poss, target, batch_size=1):
    #to be implemented
    accuracy = 0
    return accuracy

def accuracy_threshold(prediction_poss, target, batch_size=1, threshold = 0.05):
    # to be implemented
    accuracy = 0
    return accuracy