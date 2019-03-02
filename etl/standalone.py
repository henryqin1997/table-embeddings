#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import locale
from collections import defaultdict
from operator import itemgetter
from .table import Table
from .table import satisfy_variants
from .tagger import st, tag_to_index
from .feature_identifier import nst_encoding, nst_decoding, identify_nst, identify_date
from numerical_space.load import is_numeric

wordlist_json = './data/wordlist_v6_index.json'
wordlist = json.load(open(wordlist_json))


def generate_input_target(data):
    if not satisfy_variants(data):
        raise ValueError('Table format not supported')

    table = Table(data)
    column_num = len(table.get_header())
    attributes = table.get_attributes()

    nst = [0] * 10
    for idx in range(min(column_num, 10)):
        attribute = attributes[idx]
        nst_count = defaultdict(int)
        for value in attribute:
            nst_count[nst_encoding(identify_nst(value))] += 1
        nst_count = dict(sorted(nst_count.items(), key=itemgetter(1), reverse=True))
        nst_max, nst_max_count = list(nst_count.items())[0]
        nst[idx] = nst_max if nst_max_count > sum(nst_count.values()) * 0.5 else 0
    for idx in range(10):
        nst[idx] = [int(value) for value in reversed(nst_decoding(nst[idx]))]
    nst = np.array(nst, dtype=int)

    ner = table.generate_ner_matrix(st, tag_to_index).transpose()

    is_date = np.zeros(10, dtype=int)
    for idx in range(min(column_num, 10)):
        attribute = attributes[idx]
        attribute_is_date = 0
        for value in attribute:
            if identify_date(value):
                attribute_is_date = 1
                break
        is_date[idx] = attribute_is_date
    is_date = is_date.reshape((10, 1))

    is_numeric_input = np.zeros(10, dtype=int)
    is_float_input = np.zeros(10, dtype=int)
    is_ordered_input = [[0, 0]] * 10
    for idx in range(min(column_num, 10)):
        if nst[idx].tolist() == [0, 0, 1] or nst[idx].tolist() == [0, 1, 1]:
            attribute = attributes[idx]
            if all(list(map(lambda n: is_numeric(n) or n.upper() in ['', 'NA', 'N/A'], attribute))):
                is_numeric_input[idx] = 1
                is_float_input[idx] = int('.' in ''.join(attribute))
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
                values = np.array(list(map(locale.atof, filter(is_numeric, attribute))))
                # [0,0]: random, [1,0]: desc, [0,1]: asc
                is_ordered_input[idx] = [0, 1] if np.all(np.diff(values) > 0) else \
                    [1, 0] if np.all(np.diff(values) < 0) else [0, 0]
    is_numeric_input = is_numeric_input.reshape((10, 1))
    is_float_input = is_float_input.reshape((10, 1))
    is_ordered_input = np.array(is_ordered_input, dtype=int)

    is_empty_column = (np.linspace(0, 9, 10, dtype=int) >= column_num).astype(int).reshape((10, 1))

    input = np.concatenate((nst, ner, is_date, is_numeric_input, is_float_input, is_ordered_input, is_empty_column),
                           axis=1).astype(int)

    label = table.generate_wordlist_matrix(wordlist).transpose().astype(int)

    return input, label


if __name__ == '__main__':
    data = json.load(
        open('data/train_100_sample/0/1438042988061.16_20150728002308-00106-ip-10-236-191-2_173137181_0.json'))
    generate_input_target(data)
