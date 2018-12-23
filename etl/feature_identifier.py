from dateutil.parser import parse
import json
import os
import re
import numpy
from .table import Table
from collections import defaultdict

webtables_dir = './webtables'
training_data_dir = './data/train_100_sample'
training_files_json = './data/training_files_100_sample.json'
training_files = json.load(open(training_files_json))


def identify_date(value):
    try:
        parse(value, fuzzy_with_tokens=True)
        return True
    except ValueError:
        return False


def identify_number(value):
    return bool(re.search(r'\d+', value))


def identify_symbol(value):
    return bool(re.search(r'[^a-zA-Z0-9]+', value))


def identify_text(value):
    return bool(re.search(r'[a-zA-Z]+', value))


def identify_nst(value):
    return [identify_number(value), identify_symbol(value), identify_text(value)]


def nst_encoding(nst):
    return (4 if nst[0] else 0) + (2 if nst[1] else 0) + (1 if nst[2] else 0)


def identify_features(training_files):
    for training_file in training_files:
        data = json.load(open(os.path.join(webtables_dir, training_file), encoding='utf-8'))
        table = Table(data)
        print(training_file)
        for attribute in table.get_attributes():
            nst_count = defaultdict(int)
            for value in attribute:
                nst_count[nst_encoding(identify_nst(value))] += 1
                if nst_encoding(identify_nst(value)) == 0:
                    print(value)
            print(nst_count)
        if not os.path.exists(os.path.join(training_data_dir, os.path.dirname(training_file))):
            os.makedirs(os.path.join(training_data_dir, os.path.dirname(training_file)))
        basename = training_file.rstrip('.json')
        numpy.savetxt(os.path.join(training_data_dir, '{}_nst.csv'.format(basename)),
                      [[1, 2, 3, 4], [5, 6, 7, 8]], fmt='%i', delimiter=",")


if __name__ == '__main__':
    identify_features(training_files)
