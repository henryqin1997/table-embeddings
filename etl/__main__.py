#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import re
import numpy
import operator
from collections import defaultdict
from multiprocessing import Process
from .table import Table
from .table import satisfy_variants
from .tagger import st, tag_to_index

training_data_dir = './data/train_100_sample'
webtables_dir = './webtables'
num_processors = int(sys.argv[1])
training_files_json = './data/training_files_100_sample.json'
wordlist_json = './data/wordlist_v6_index.json'

# # For sample random label
# training_data_dir = './data/sample_random_table_test'
# webtables_dir = './data/sample_random_table'
# num_processors = int(sys.argv[1])
# training_files_json = './data/testing_files_random_table.json'
# wordlist_json = './data/wordlist.json'


def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

def conduct_etl(training_files):
    wordlist = json.load(open(wordlist_json))
    for training_file in training_files:
        data = json.load(open(os.path.join(webtables_dir, training_file), encoding='utf-8'))
        if satisfy_variants(data):
            table = Table(data)
            # # Filter table with labels <= 10
            # if len(table.get_header()) <= 10:
            print(training_file)
            if not os.path.exists(os.path.join(training_data_dir, os.path.dirname(training_file))):
                os.makedirs(os.path.join(training_data_dir, os.path.dirname(training_file)))
            basename = training_file.rstrip('.json')
            json.dump(table.data,
                      open(os.path.join(training_data_dir, training_file), 'w+'),
                      indent=4)
            numpy.savetxt(os.path.join(training_data_dir, '{}_ner.csv'.format(basename)),
                          table.generate_ner_matrix(st, tag_to_index), fmt='%i', delimiter=",")
            numpy.savetxt(os.path.join(training_data_dir, '{}_wordlist.csv'.format(basename)),
                          table.generate_wordlist_matrix(wordlist), fmt='%i', delimiter=",")


def main():
    # # Generate wordlist and training files list
    # json.dump(generate_wordlist(), open('data/wordlist.json', 'w+'), indent=4)
    # json.dump(list_training_files(), open('data/training_files.json', 'w+'), indent=4)
    # return
    training_files = json.load(open(training_files_json))
    for training_files_chunk in chunkify(training_files,num_processors):
        Process(target=conduct_etl, args=(training_files_chunk,)).start()


if __name__ == '__main__':
    main()
