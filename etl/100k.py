import json
import os
import random
from .table import Table, satisfy_variants

random.seed(10)

files_json_1m = 'data/1m_files.json'
all_files = []

if __name__ == '__main__':
    files_1m = json.load(open(files_json_1m))
    random.shuffle(files_1m)
    all_files = files_1m[::10]
    json.dump(all_files, open('data/100k_files.json', 'w+'), indent=4)
