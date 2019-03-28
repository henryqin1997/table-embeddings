import json
from collections import defaultdict
from operator import itemgetter
from .generator import tables

if __name__ == '__main__':
    key_column_stats = defaultdict(lambda: defaultdict(int))

    for table in tables('data/training_files_100_sample.json', prefix='data/train_100_sample'):
        index = table.get_data()['keyColumnIndex']
        if index >= 0:
            key_column_stats[table.get_header()[index]][index] += 1

    for k, v in key_column_stats.items():
        key_column_stats[k] = dict(sorted(key_column_stats[k].items(), key=itemgetter(1), reverse=True))

    key_column_stats = dict(sorted(key_column_stats.items(), key=lambda item: sum(item[1].values()), reverse=True))

    json.dump(key_column_stats, open('data/key_column_stats.json', 'w+'), indent=4)
