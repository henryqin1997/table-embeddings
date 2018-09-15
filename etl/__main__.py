#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy
from .table import Table
from .tagger import st, tag_to_index

data_dir = './data'


def satisfy_variants(data):
    return data['tableType'] == 'RELATION' and data['hasHeader'] and data['headerRowIndex'] >= 0 and data[
        'tableOrientation'] in ['HORIZONTAL', 'VERTICAL']


def main():
    tables = []
    with open(os.path.join(data_dir, 'sample'), encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if satisfy_variants(data) and len(data['relation']) in range(10, 20) and len(data['relation'][0]) in range(
                    10, 20):
                tables.append(Table(data))

    for table in tables:
        if table.get_data_md5() == '0abc1124b6766c9bf281982a4e6adc5e':
            json.dump(data,
                      open(os.path.join(data_dir, 'table-{}.json'.format(table.get_data_md5())), 'w+'),
                      indent=4)
            print(table.get_header())
            print(table.get_entities())
            print(table.get_attributes())
            print(table.generate_ner_matrix(st, tag_to_index))
            print(table.generate_wordlist_matrix({'Date': 0, 'Name': 1, 'Opponent': 2}))
            print()


if __name__ == '__main__':
    main()
