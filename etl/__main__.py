#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

data_dir = './data'


def get_string_md5(string):
    import hashlib
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def main():
    with open(os.path.join(data_dir, 'sample'), encoding='utf-8') as f:
        for line in f:
            table = json.loads(line)
            if len(table['relation']) > 15:
                json.dump(table, open(os.path.join(data_dir, 'table-{}.json'.format(get_string_md5(line))), 'w+'),
                          indent=4)


if __name__ == '__main__':
    main()
