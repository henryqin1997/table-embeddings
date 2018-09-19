#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy
import operator
from collections import defaultdict
from .table import Table
from .table import satisfy_variants
from .tagger import st, tag_to_index

data_dir = './data'
wordlist_raw = './hadoop/output2/part-00000'


def generate_wordlist(raw_file):
    wordlist = {}
    index = 0
    with open(raw_file) as f:
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


def main():
    json.dump(generate_wordlist(wordlist_raw), open('data/wordlist.json', 'w+'), indent=4)
    return
    tables = []
    with open(os.path.join(data_dir, 'sample'), encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if satisfy_variants(data):
                table = Table(data)
                # # Filter table with labels <= 10
                # if len(table.get_header()) <= 10:
                tables.append(Table(data))

    wordlist = defaultdict(int)
    for table in tables:
        for label in table.get_header():
            label = label.lower().strip()
            if len(label) > 2:
                wordlist[label.lower().strip()] += 1
    # Sort wordlist by frequency
    wordlist = dict(sorted(wordlist.items(), key=operator.itemgetter(1), reverse=True))
    with open(os.path.join(data_dir, 'wordlist.json'), 'w+') as f:
        json.dump(wordlist, f, indent=4)

    # Filter table with labels <= 10
    tables = list(filter(lambda table: len(table.get_header()) <= 10, tables))

    for table in tables:
        if table.get_data_md5() == '0abc1124b6766c9bf281982a4e6adc5e':
            json.dump(table.data,
                      open(os.path.join(data_dir, 'table-{}.json'.format(table.get_data_md5())), 'w+'),
                      indent=4)
            print(table.get_header())
            print(table.get_entities())
            print(table.get_attributes())
            print(table.generate_ner_matrix(st, tag_to_index))
            print(table.generate_wordlist_matrix({'date': 0, 'name': 1, 'opponent': 2}))
            print()


if __name__ == '__main__':
    main()
