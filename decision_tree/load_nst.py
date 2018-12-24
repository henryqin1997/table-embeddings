import json
import os
import numpy
from etl import Table

training_data_dir = 'data/train'
training_files_json = 'data/training_files.json'
training_files = json.load(open(training_files_json))


def load_nst_major(batch_size, batch_index):
    batch_files = training_files[batch_size * batch_index:batch_size * (batch_index + 1)]
    batch_files_nst = list(map(lambda batch_file: batch_file.rstrip('.json') + '_nst.csv', batch_files))
    batch_files_wordlist = list(map(lambda batch_file: batch_file.rstrip('.json') + '_wordlist.csv', batch_files))
    inputs = numpy.array(
        [list(map(to_int, numpy.genfromtxt(os.path.join(training_data_dir, batch_file_nst), delimiter=',')[0])) for
         batch_file_nst in batch_files_nst])
    targets = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_wordlist), delimiter=',') for batch_file_wordlist
         in batch_files_wordlist])

    targets_transformed = []

    for i in range(len(targets)):
        table = Table(json.load(open(os.path.join(training_data_dir, batch_files[i]))))
        column_num = len(table.get_header())
        target = targets[i]

        targets_transformed.append(
            numpy.array([index_of(list(map(lambda num: int(round(num)), row)), 1) if idx < column_num else -1 for
                         idx, row in enumerate(target.transpose())]).transpose())
    return inputs, numpy.array(targets_transformed)


def load_nst_max(batch_size, batch_index):
    batch_files = training_files[batch_size * batch_index:batch_size * (batch_index + 1)]
    batch_files_nst = list(map(lambda batch_file: batch_file.rstrip('.json') + '_nst.csv', batch_files))
    batch_files_wordlist = list(map(lambda batch_file: batch_file.rstrip('.json') + '_wordlist.csv', batch_files))
    inputs = numpy.array(
        [list(map(to_int, numpy.genfromtxt(os.path.join(training_data_dir, batch_file_nst), delimiter=',')[1])) for
         batch_file_nst in batch_files_nst])
    targets = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_wordlist), delimiter=',') for batch_file_wordlist
         in batch_files_wordlist])

    targets_transformed = []

    for i in range(len(targets)):
        table = Table(json.load(open(os.path.join(training_data_dir, batch_files[i]))))
        column_num = len(table.get_header())
        target = targets[i]

        targets_transformed.append(
            numpy.array([index_of(list(map(lambda num: int(round(num)), row)), 1) if idx < column_num else -1 for
                         idx, row in enumerate(target.transpose())]).transpose())
    return inputs, numpy.array(targets_transformed)


def load_nst_overall(batch_size, batch_index):
    batch_files = training_files[batch_size * batch_index:batch_size * (batch_index + 1)]
    batch_files_nst = list(map(lambda batch_file: batch_file.rstrip('.json') + '_nst.csv', batch_files))
    batch_files_wordlist = list(map(lambda batch_file: batch_file.rstrip('.json') + '_wordlist.csv', batch_files))
    inputs = numpy.array(
        [list(map(to_int, numpy.genfromtxt(os.path.join(training_data_dir, batch_file_nst), delimiter=',')[2])) for
         batch_file_nst in batch_files_nst])
    targets = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_wordlist), delimiter=',') for batch_file_wordlist
         in batch_files_wordlist])

    targets_transformed = []

    for i in range(len(targets)):
        table = Table(json.load(open(os.path.join(training_data_dir, batch_files[i]))))
        column_num = len(table.get_header())
        target = targets[i]

        targets_transformed.append(
            numpy.array([index_of(list(map(lambda num: int(round(num)), row)), 1) if idx < column_num else -1 for
                         idx, row in enumerate(target.transpose())]).transpose())
    return inputs, numpy.array(targets_transformed)


def load_nst_majo(batch_size, batch_index):
    '''To be implemented'''


def load_nst_maxo(batch_size, batch_index):
    '''To be implemented'''


def load_nst_mm(batch_size, batch_index):
    '''To be implemented'''


def load_nst_mmo(batch_size, batch_index):
    '''To be implemented'''


def index_of(l, n):
    try:
        return list(l).index(n)
    except ValueError:
        return -1


def to_int(n):
    return int(round(n))


if __name__ == '__main__':
    print(load_nst_major(50, 2060))
