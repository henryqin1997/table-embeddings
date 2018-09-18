#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
from etl import Table
from etl import satisfy_variants

for line in sys.stdin:
    data = json.loads(line)
    if satisfy_variants(data):
        table = Table(data)
        words = table.get_header()
        for word in words:
            word = word.lower().strip()
            if len(word) > 2:
                print('{}\t{}'.format(word, 1))
