import json
import os
import random
from .table import Table, satisfy_variants

random.seed(10)

files_json_1m = 'data/1m_files.json'
all_files = []

if __name__ == '__main__':
    files_1m = json.load(open(files_json_1m))
    for file in files_1m:
        if random.random() < 0.1:
            all_files.append(file)
    json.dump(all_files, open('data/100k_files.json', 'w+'), indent=4)
