import json
import os
import numpy

training_data_dir = '../data/train'
training_files_json = '../data/training_files_filtered.json'
training_files = json.load(open(training_files_json))
tag_to_index = {'LOCATION': 0, 'PERSON': 1, 'ORGANIZATION': 2, 'MONEY': 3, 'PERCENT': 4, 'DATE': 5, 'TIME': 6}


def one_hot(row):
    assert len(row) > 0
    row_sum = int(round(sum(numpy.array([(2 ** i) * num for (i, num) in enumerate(row)]))))
    row_converted = numpy.zeros(2 ** len(row))
    assert row_sum < len(row_converted)
    row_converted[row_sum] = 1
    return row_converted


def load_data(batch_size, batch_index=0):
    # load training data from file, to be implemented
    # put size number of data into one array
    # start from batch_index batch
    batch_files = training_files[batch_size * batch_index:batch_size * (batch_index + 1)]
    batch_files_ner = list(map(lambda batch_file: batch_file.rstrip('.json') + '_ner.csv', batch_files))
    batch_files_wordlist = list(map(lambda batch_file: batch_file.rstrip('.json') + '_wordlist.csv', batch_files))
    inputs = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_ner), delimiter=',') for batch_file_ner in
         batch_files_ner])
    targets = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_wordlist), delimiter=',') for batch_file_wordlist
         in batch_files_wordlist])

    inputs_transformed = []
    targets_transformed = []

    # Use One Hot Encoding and remove column with all zeros
    for i in range(len(inputs)):
        input = inputs[i]
        target = targets[i]
        assert len(input) == len(tag_to_index)

        input = numpy.array([one_hot(row) for row in input.transpose()])
        target = target.transpose()

        input_transformed = numpy.zeros(input.shape)
        target_transformed = numpy.zeros(target.shape)

        current_index = 0
        for row in input:
            if row[0] == 0:
                input_transformed[current_index] = input[current_index]
                target_transformed[current_index] = target[current_index]
                current_index += 1

        inputs_transformed.append(input_transformed.transpose())
        targets_transformed.append(target_transformed.transpose())
    return numpy.array(inputs_transformed), numpy.array(targets_transformed)


##########################333#3#
# evaluation of model:
# 1. accuracy of prediction of label over target     #correct prediction/#targetlabel
# 2. accuracy of prediction of label over target when other doesn't count
#   #correct prediction(no 'other')/#targetlabel(no other)

def accuracy(prediction, target, batch_size=1):  # to be implemented
    if batch_size>1:
        total_num = 0
        correct_num = 0.
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                correct_num = correct_num+float(prediction[batch_index][:,col_index].dot(target[batch_index][:,col_index]))
                total_num = total_num + int(sum(target[batch_index][:, col_index]))
        return correct_num / col_size
    else:
        total_num = 0
        correct_num = 0
        col_size = target.shape[1]
        for col_index in range(col_size):
            correct_num = correct_num + float(prediction[:, col_index].dot(target[:, col_index]))
            total_num = total_num + int(sum(target[:, col_index]))
        return correct_num / col_size


def accuracy_no_other(prediction, target, batch_size=1):  # to be implemented
    if batch_size>1:
        total_num = 0
        correct_num = 0
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                correct_num=correct_num+float(prediction[batch_index][:-1,col_index].dot(target[batch_index][:-1,col_index]))
                total_num = total_num + int(sum(target[batch_index][:-1, col_index]))
        return correct_num / total_num
    else:
        total_num = 0
        correct_num = 0
        col_size = target.shape[1]
        for col_index in range(col_size):
            correct_num = correct_num + float(prediction[:-1, col_index].dot(target[:-1, col_index]))
            total_num = total_num + int(sum(target[:-1, col_index]))
        return correct_num / total_num

def accuracy_possibility(prediction_poss, target, batch_size=1):
    #to be implemented
    return accuracy(prediction_poss,target,batch_size)

def accuracy_threshold(prediction_poss, target, batch_size=1, threshold = 0.05):
    # to be implemented
    accuracy = 0
    if batch_size>1:
        total_num = 0
        correct_num = 0
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                prob=float(prediction_poss[batch_index][:, col_index].dot(target[batch_index][:, col_index]))
                if prob<threshold:
                    prob=0
                correct_num=correct_num+prob
                total_num = total_num + int(sum(prediction_poss[batch_index][:, col_index]))
        accuracy=correct_num / total_num
    else:
        total_num = 0
        correct_num = 0
        col_size = target.shape[1]
        print(col_size)
        for col_index in range(col_size):
            prob=float(prediction_poss[:, col_index].dot(target[:, col_index]))
            if prob < threshold:
                prob = 0
            correct_num = correct_num + prob
            total_num = total_num + int(sum(prediction_poss[:, col_index]))
        accuracy=correct_num / total_num
    return accuracy