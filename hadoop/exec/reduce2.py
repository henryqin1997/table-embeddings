#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import collections

WORDDICT = {}
for line in sys.stdin:
    line = line.strip()
    i = line.find('\t')
    count = int(line[:i])
    word = line[i + 1:]
    if count in WORDDICT:
        WORDDICT[count].append(word)
    else:
        WORDDICT[count] = [word]

SORTEDDICT = collections.OrderedDict(sorted(WORDDICT.items(), reverse=True))
for key in SORTEDDICT:
    for word in SORTEDDICT[key]:
        print('{}\t{}'.format(key, word))
