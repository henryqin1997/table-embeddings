import sys
import json
import os
from etl.table import Table

if __name__ == '__main__':
    label = sys.argv[1]
    files = json.load(open('data/training_files.json'))
    for file in files:
        data = json.load(open(os.path.join('data/train', file)))
        table = Table(data)
        if label in table.get_header():
            print(file)
