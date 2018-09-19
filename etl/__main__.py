#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
import numpy
import operator
from collections import defaultdict
from .table import Table
from .table import satisfy_variants
from .tagger import st, tag_to_index

data_dir = './data'
webtables_dir = './webtables'
wordlist_raw = './hadoop/output2/part-00000'
tree_dir = './data/tree'
num_folders = 51
training_files_json = './data/training_files.json'
wordlist_json = './data/wordlist.json'


def generate_wordlist():
    wordlist = {}
    index = 0
    with open(wordlist_raw) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            i = line.find('\t')
            count = int(line[:i])
            word = line[i + 1:]
            if count > 1:
                wordlist[word] = index
                index += 1
    return wordlist


def training_file_filter(file, i):
    i = str(i)
    return bool(re.match(r'(.+?){}-ip(.+?)25.json'.format(i.zfill(2)), file))


def list_training_files():
    all_files = []
    for i in range(num_folders):
        with open(os.path.join(tree_dir, str(i))) as f:
            files = f.readlines()
        files = list(
            map(lambda file: os.path.join(str(i), file.strip()),
                filter(lambda file: training_file_filter(file, i), files)))
        all_files += files
    return all_files


def main():
    # # Generate wordlist and training files list
    # json.dump(generate_wordlist(), open('data/wordlist.json', 'w+'), indent=4)
    # json.dump(list_training_files(), open('data/training_files.json', 'w+'), indent=4)
    # return

    training_files = json.load(open(training_files_json))
    wordlist = json.load(open(wordlist_json))
    tables = []
    for training_file in training_files:
        with open(os.path.join(webtables_dir, training_file), encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if satisfy_variants(data):
                    tables.append(Table(data))

    # Filter table with labels <= 10
    tables = list(filter(lambda table: len(table.get_header()) <= 10, tables))

    for table in tables:
        md5 = table.get_data_md5()
        print(md5)
        json.dump(table.data,
                  open(os.path.join(data_dir, 'train', '{}_table.json'.format(md5)), 'w+'),
                  indent=4)
        table.generate_ner_matrix(st, tag_to_index).dump(os.path.join(data_dir, 'train', '{}_ner.mat'.format(md5)))
        table.generate_wordlist_matrix(wordlist).dump(os.path.join(data_dir, 'train', '{}_wordlist.mat'.format(md5)))
        # print(table.get_header())
        # print(table.get_entities())
        # print(table.get_attributes())
        # print(table.generate_ner_matrix(st, tag_to_index))
        # print(table.generate_wordlist_matrix(wordlist))


if __name__ == '__main__':
    main()
