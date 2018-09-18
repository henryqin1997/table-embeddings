#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import numpy


def satisfy_variants(data):
    return data['tableType'] == 'RELATION' and data['hasHeader'] and data['headerRowIndex'] >= 0 and data[
        'tableOrientation'] in ['HORIZONTAL', 'VERTICAL']


def extract_header(data):
    if data['tableOrientation'] == 'HORIZONTAL':
        return numpy.array([item[data['headerRowIndex']] for item in data['relation']])
    else:
        return numpy.array(data['relation'][data['headerRowIndex']])


for line in sys.stdin:
    data = json.loads(line)
    if satisfy_variants(data):
        words = extract_header(data)
        for word in words:
            word = word.lower().strip()
            if len(word) > 2:
                print('{}\t{}'.format(word, 1))
