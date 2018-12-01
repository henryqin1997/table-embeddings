import json
import os
import random
from .table import Table, satisfy_variants

random.seed(10)

num_folders = 51
min_num_rows = 5
all_files = []

if __name__ == '__main__':
    for folder in range(num_folders):
        folder = str(folder)
        print(folder)
        for file in os.listdir(os.path.join('webtables', folder)):
            if random.random() < 0.035:
                data = json.load(open(os.path.join('webtables', folder, file)))
                if satisfy_variants(data):
                    table = Table(data)
                    if len(table.get_entities()) >= min_num_rows:
                        all_files.append(os.path.join(folder, file))
    json.dump(all_files, open('data/1m_files.json', 'w+'), indent=4)
