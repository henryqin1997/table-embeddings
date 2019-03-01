import sys
import json
import os
from shutil import copyfile, rmtree
from etl.table import Table

if __name__ == '__main__':
    rmtree('data/example')
    label = sys.argv[1]
    files = json.load(open('data/training_files.json'))
    for file in files:
        data = json.load(open(os.path.join('data/train', file)))
        table = Table(data)
        if label in [header.lower() for header in table.get_header()]:
            dir_name, _ = os.path.split(file)
            os.makedirs(os.path.join('data/example', dir_name), exist_ok=True)
            copyfile(os.path.join('data/train', file), os.path.join('data/example', file))
            print(file)
