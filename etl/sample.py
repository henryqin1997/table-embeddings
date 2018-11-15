'''
Build a sample with random distribution of labels
'''
import sys
import random

seed = sys.argv[1]
random.seed(int(seed))
import json
import os
import operator
import shutil
from collections import defaultdict
from .table import Table, satisfy_variants

wordlist_json = 'data/wordlist_v4.json'
input_dir = 'data/input'
output_dir_random_label = 'data/sample_random_label'
testing_filelist_json_random_label = 'data/testing_files_random_label.json'
output_dir_random_table = 'data/sample_random_table'
testing_filelist_json_random_table = 'data/testing_files_random_table.json'

try:
    shutil.rmtree(os.path.join(output_dir_random_label, seed))
except FileNotFoundError:
    pass
os.makedirs(os.path.join(output_dir_random_label, seed))
total_table_num = 115859


def generate_random_label_filelist():
    folders = sorted(list(filter(lambda item: item.isdigit(), os.listdir(output_dir_random_label))))
    filelist = []
    for folder in folders:
        filelist += [os.path.join(folder, filename) for filename in
                     sorted(list(filter(lambda item: item.endswith('.json') and not item.endswith('_activate.json'),
                                        os.listdir(os.path.join(output_dir_random_label, folder)))))]
    return filelist

def generate_random_table_filelist():
    folders = sorted(list(filter(lambda item: item.isdigit(), os.listdir(output_dir_random_table))))
    filelist = []
    for folder in folders:
        filelist += [os.path.join(folder, filename) for filename in
                     sorted(list(filter(lambda item: item.endswith('.json') and not item.endswith('_activate.json'),
                                        os.listdir(os.path.join(output_dir_random_table, folder)))))]
    return filelist


def random_label():
    wordlist = json.load(open(wordlist_json))
    word_prob = dict([(item[0], float(sys.argv[2]) / item[1]) for item in wordlist.items()])
    word_count = defaultdict(int)

    lines = []
    files = os.listdir(input_dir)
    for file in files:
        with open(os.path.join(input_dir, file), encoding='utf-8') as f:
            lines += f.readlines()
    random.shuffle(lines)

    table_num = 0

    for line in lines:
        line = line.strip()
        data = json.loads(line)
        if satisfy_variants(data):
            table = Table(data)
            words = table.get_header()
            activate = [0] * 10
            for idx, word in enumerate(words[:10]):
                word = word.lower().strip()
                if word in word_prob and random.random() < word_prob[word]:
                    activate[idx] = 1
                    word_count[word] += 1
            if any(item for item in activate):
                table_num += 1

                filename = str(table_num).zfill(4)
                json.dump(data, open(os.path.join(output_dir_random_label, seed, filename + '.json'), 'w+'))
                json.dump(activate,
                          open(os.path.join(output_dir_random_label, seed, filename + '_activate.json'), 'w+'))
    print(table_num)
    # print(sorted(word_count.items(), key=operator.itemgetter(1)))


def random_table():
    lines = []
    files = os.listdir(input_dir)
    for file in files:
        with open(os.path.join(input_dir, file), encoding='utf-8') as f:
            lines += f.readlines()
    random.shuffle(lines)

    for i, line in enumerate(lines[:3000]):
        line = line.strip()
        data = json.loads(line)
        filename = str(i).zfill(4)
        json.dump(data, open(os.path.join(output_dir_random_table, seed, filename + '.json'), 'w+'))


if __name__ == '__main__':
    random_table()
    json.dump(generate_random_table_filelist(), open(testing_filelist_json_random_table, 'w+'), indent=4)
