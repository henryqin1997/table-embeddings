#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import collections

WORDDICT = {}
for line in sys.stdin:
    i = line.rfind('\t')
    word = line[:i]
    if word in WORDDICT:
        WORDDICT[word] += 1
    else:
        WORDDICT[word] = 1

for key in WORDDICT:
    print('{}\t{}'.format(key, WORDDICT[key]))
