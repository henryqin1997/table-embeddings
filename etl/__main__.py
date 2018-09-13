#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

data_dir = './data'


def satisfy_variants(data):
    return data['tableType'] == 'RELATION' and data['hasHeader'] and data['headerRowIndex'] >= 0 and data[
        'tableOrientation'] in ['HORIZONTAL', 'VERTICAL']


class Table():
    def __init__(self, data):
        assert satisfy_variants(data)
        self.data = data

    def extract_header(self):
        if self.data['tableOrientation'] == 'HORIZONTAL':
            return [item[self.data['headerRowIndex']] for item in self.data['relation']]
        else:
            return self.data['relation'][self.data['headerRowIndex']]

    def extract_entities(self):
        relation = self.data['relation']
        if self.data['tableOrientation'] == 'HORIZONTAL':
            return [[item[i] for item in relation] for (i, val) in enumerate(relation[0]) if
                    i != self.data['headerRowIndex']]
        else:
            return [relation[i] for (i, val) in enumerate(relation) if i != self.data['headerRowIndex']]

    def get_data_md5(self):
        import hashlib
        return hashlib.md5(json.dumps(self.data).encode('utf-8')).hexdigest()


def main():
    with open(os.path.join(data_dir, 'sample'), encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if satisfy_variants(data) and len(data['relation']) in range(10, 20) and len(data['relation'][0]) in range(
                    10, 20):
                table = Table(data)
                json.dump(data,
                          open(os.path.join(data_dir, 'table-{}.json'.format(table.get_data_md5())), 'w+'),
                          indent=4)
                print(table.extract_header())
                print(table.extract_entities())
                print()


if __name__ == '__main__':
    main()
