import json
import os
import numpy
from collections import defaultdict
from etl import Table

training_data_dir = 'data/train'
training_files_json = 'data/training_files_filtered.json'
training_files = json.load(open(training_files_json))
testing_data_dir = 'data/sample_random_label_train'
activate_data_dir = 'data/sample_random_label'
testing_files_json = 'data/testing_files_random_label.json'
testing_files_json = 'data/testing_files_random_label_sample.json'
testing_files = [[y for y in json.load(open(testing_files_json)) if y[0] == str(x)] for x in range(10)]
tag_to_index = {'LOCATION': 0, 'PERSON': 1, 'ORGANIZATION': 2, 'MONEY': 3, 'PERCENT': 4, 'DATE': 5, 'TIME': 6}