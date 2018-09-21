#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

for line in sys.stdin:
    line = line.strip()
    i = line.rfind('\t')
    word = line[:i]
    count = line[i + 1:]
    print(count + '\t' + word)
