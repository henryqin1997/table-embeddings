import json
import os
import numpy
from etl import Table

training_data_dir = 'data/train'
training_files_json = 'data/training_files_filtered.json'
training_files = json.load(open(training_files_json))
testing_data_random_label_dir = 'data/sample_random_label_test'
activate_data_random_label_dir = 'data/sample_random_label'
testing_files_random_label_json = 'data/testing_files_random_label.json'
testing_files_random_label = [[y for y in json.load(open(testing_files_random_label_json)) if y[0] == str(x)] for x in range(10)]
testing_data_random_table_dir = 'data/sample_random_table_test'
activate_data_random_table_dir = 'data/sample_random_table'
testing_files_random_table_json = 'data/testing_files_random_table.json'
testing_files_random_table = [[y for y in json.load(open(testing_files_random_table_json)) if y[0] == str(x)] for x in range(1)]
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

    # Use One Hot Encoding
    for i in range(len(inputs)):
        input = inputs[i]
        target = targets[i]
        assert len(input) == len(tag_to_index)

        inputs_transformed.append(numpy.array([one_hot(row) for row in input.transpose()]).transpose())
        targets_transformed.append(target)
    return numpy.array(inputs_transformed), numpy.array(targets_transformed)


def indexOf(l, n):
    try:
        return list(l).index(n)
    except ValueError:
        return -1


def load_sample_random_label(sample_index, batch_size, batch_index):
    # load testing data of sample with random labels
    # put size number of data into one array
    # start from batch_index batch
    result = []
    batch_files = testing_files_random_label[sample_index][batch_size * batch_index:batch_size * (batch_index + 1)]
    for batch_file in batch_files:
        table = Table(json.load(open(os.path.join(testing_data_random_label_dir, batch_file))))
        column_num = len(table.get_header())
        batch_file_ner = batch_file.rstrip('.json') + '_ner.csv'
        batch_file_wordlist = batch_file.rstrip('.json') + '_wordlist.csv'
        batch_file_activate = batch_file.rstrip('.json') + '_activate.json'
        input = numpy.genfromtxt(os.path.join(testing_data_random_label_dir, batch_file_ner), delimiter=',').transpose()
        target = numpy.genfromtxt(os.path.join(testing_data_random_label_dir, batch_file_wordlist), delimiter=',').transpose()
        activate = json.load(open(os.path.join(activate_data_random_label_dir, batch_file_activate)))

        input_transformed = [
            int(round(sum(numpy.array([(2 ** i) * num for (i, num) in enumerate(row)]))))
            if idx < column_num else -1 for idx, row in enumerate(input)]
        target_transformed = [indexOf(list(map(lambda num: int(round(num)), row)), 1) if idx < column_num else -1 for
                              idx, row in enumerate(target)]
        activate_transformed = [num if idx < column_num else -1 for idx, num in enumerate(activate)]

        result.append([input_transformed, target_transformed, activate_transformed])
    return result


def load_sample_random_table(sample_index, batch_size, batch_index):
    # load testing data of sample with random tables
    # put size number of data into one array
    # start from batch_index batch
    result = []
    batch_files = testing_files_random_table[sample_index][batch_size * batch_index:batch_size * (batch_index + 1)]
    for batch_file in batch_files:
        table = Table(json.load(open(os.path.join(testing_data_random_table_dir, batch_file))))
        column_num = len(table.get_header())
        batch_file_ner = batch_file.rstrip('.json') + '_ner.csv'
        batch_file_wordlist = batch_file.rstrip('.json') + '_wordlist.csv'
        input = numpy.genfromtxt(os.path.join(testing_data_random_table_dir, batch_file_ner), delimiter=',').transpose()
        target = numpy.genfromtxt(os.path.join(testing_data_random_table_dir, batch_file_wordlist), delimiter=',').transpose()

        input_transformed = [
            int(round(sum(numpy.array([(2 ** i) * num for (i, num) in enumerate(row)]))))
            if idx < column_num else -1 for idx, row in enumerate(input)]
        target_transformed = [indexOf(list(map(lambda num: int(round(num)), row)), 1) if idx < column_num else -1 for
                              idx, row in enumerate(target)]

        result.append([input_transformed, target_transformed])
    return result