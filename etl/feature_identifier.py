from dateutil.parser import parse
import json
import os
import re
import numpy
from .table import Table
from collections import defaultdict
from operator import itemgetter
from functools import reduce

webtables_dir = './webtables'
training_data_dir = './data/train'
training_files_json = './data/training_files.json'
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
        major = []
        max = []
        overall = []

        for attribute in table.get_attributes():
            nst_count = defaultdict(int)
            for value in attribute:
                nst_count[nst_encoding(identify_nst(value))] += 1
            nst_count = dict(sorted(nst_count.items(), key=itemgetter(1), reverse=True))
            nst_max, nst_max_count = list(nst_count.items())[0]
            major.append(nst_max if nst_max_count > sum(nst_count.values()) * 0.5 else 0)
            max.append(nst_max)
            overall.append(reduce(lambda a, b: a | b, nst_count.keys()))

        if not os.path.exists(os.path.join(training_data_dir, os.path.dirname(training_file))):
            os.makedirs(os.path.join(training_data_dir, os.path.dirname(training_file)))
        basename = training_file.rstrip('.json')
        numpy.savetxt(os.path.join(training_data_dir, '{}_nst.csv'.format(basename)),
                      [major, max, overall], fmt='%i', delimiter=",")


def if_ordered(numbers):
    '''Input: a column of number with more than 3 values; output: boolean value whether they are ordered'''
    sign = numpy.sign(numbers[0] - numbers[1])
    for i in range(1, len(numbers) - 1):
        if numbers[i] - numbers[i + 1] != 0 and numpy.sign(numbers[i] - numbers[i + 1]) != sign:
            return False
    return True


if __name__ == '__main__':
    identify_features(training_files)
