'''
To load all numerical columns from tables.
'''
import numpy as np
import json
import os
import re
from etl import Table

training_data_dir = 'data/train_100_sample'
training_files_json = 'data/training_files_100_sample.json'
training_files = json.load(open(training_files_json))


def summary(label, column):
    """
    Input: a column of numerical values in list version
    Output: a list of label, mean, variance, min, max, is_ordered, is_real
    """
    column = np.array(column)
    values = column.astype(np.float)
    return [label, np.mean(values), np.var(values), np.min(values), np.max(values),
            1 if np.all(np.diff(values) > 0) else -1 if np.all(np.diff(values) < 0) else 0,
            '.' in ''.join(column)]


def load(batch_size, batch_index):
    """
    return summarized data for a table, ie. a list of points summary.
    """
    batch_files = training_files[batch_size * batch_index:batch_size * (batch_index + 1)]
    batch_files_nst = list(map(lambda batch_file: batch_file.rstrip('.json') + '_nst.csv', batch_files))
    batch_files_wordlist = list(map(lambda batch_file: batch_file.rstrip('.json') + '_wordlist.csv', batch_files))
    all_nst = [list(map(to_int, np.genfromtxt(os.path.join(training_data_dir, batch_file_nst), delimiter=',')[0])) for
               batch_file_nst in batch_files_nst]

    targets = np.array(
        [np.genfromtxt(os.path.join(training_data_dir, batch_file_wordlist), delimiter=',') for batch_file_wordlist
         in batch_files_wordlist])

    results = []
    for i in range(len(all_nst)):
        result = []
        table = Table(json.load(open(os.path.join(training_data_dir, batch_files[i]))))
        attributes = table.get_attributes()
        column_num = len(attributes)

        target = targets[i]
        nst = all_nst[i]

        target_transformed = [index_of(list(map(lambda num: to_int(num), row)), 1) if idx < column_num else -1 for
                              idx, row in enumerate(target.transpose())]

        for j in range(column_num):
            if j >= 10:
                break
            if nst[j] == nst_encoding([True, False, False]) or nst[j] == nst_encoding([True, True, False]):
                attribute = attributes[j]
                if all([re.match(r'^[-+]?\d*\.\d+|\d+$', value) for value in attribute]):
                    result.append(summary(target_transformed[j], attribute))
                else:
                    print(attribute)
        results.append(result)
    return results


def nst_decoding(nst):
    return list(map(lambda c: c == '1', '{0:b}'.format(nst).zfill(3)))


def nst_encoding(nst):
    return (4 if nst[0] else 0) + (2 if nst[1] else 0) + (1 if nst[2] else 0)


def index_of(l, n):
    try:
        return list(l).index(n)
    except ValueError:
        return -1


def to_int(n):
    return int(round(n))


if __name__ == '__main__':
    print(load(50, 0))
