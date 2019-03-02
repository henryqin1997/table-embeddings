#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import re
import numpy as np
import operator
from collections import defaultdict
from multiprocessing import Process
from operator import itemgetter
from .table import Table
from .table import satisfy_variants
from .tagger import st, tag_to_index
from .feature_identifier import nst_encoding, identify_nst

wordlist_json = './data/wordlist_v6_index.json'
wordlist = json.load(open(wordlist_json))


# # For sample random label
# training_data_dir = './data/sample_random_table_test'
# webtables_dir = './data/sample_random_table'
# num_processors = int(sys.argv[1])
# training_files_json = './data/testing_files_random_table.json'
# wordlist_json = './data/wordlist.json'

def generate_input_target(data):
    if satisfy_variants(data):
        table = Table(data)
        ner = table.generate_ner_matrix(st, tag_to_index)
        print(ner.shape)
        label = table.generate_wordlist_matrix(wordlist)
        print(label.shape)

        nst = np.array([-1] * 10)

        for idx, attribute in enumerate(table.get_attributes()):
            if idx == 10:
                break
            nst_count = defaultdict(int)
            for value in attribute:
                nst_count[nst_encoding(identify_nst(value))] += 1
            nst_count = dict(sorted(nst_count.items(), key=itemgetter(1), reverse=True))
            nst_max, nst_max_count = list(nst_count.items())[0]
            nst[idx] = nst_max if nst_max_count > sum(nst_count.values()) * 0.5 else 0

        print(nst)
    else:
        raise ValueError('Table format not supported')


if __name__ == '__main__':
    data = json.load(
        open('data/train_100_sample/0/1438042988061.16_20150728002308-00106-ip-10-236-191-2_173137181_0.json'))
    generate_input_target(data)
