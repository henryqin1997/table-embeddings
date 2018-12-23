from dateutil.parser import parse
import json
import os
import re

webtables_dir = './webtables'
training_data_dir = './data/train'


def identify_date(value):
    try:
        parse(value, fuzzy_with_tokens=True)
        return True
    except ValueError:
        return False


def identify_number(value):
    return bool(re.search(r'\d+', value))


def identify_symbol(value):
    return bool(re.search(r'[^a-zA-Z0-9]+', value))


def identify_text(value):
    return bool(re.search(r'[a-zA-Z]+', value))


def identify_nst(value):
    return [identify_number(value), identify_symbol(value), identify_text(value)]


# def identify_features(training_files):
#     wordlist = json.load(open(wordlist_json))
#     for training_file in training_files:
#         data = json.load(open(os.path.join(webtables_dir, training_file), encoding='utf-8'))
#         if satisfy_variants(data):
#             table = Table(data)
#             # # Filter table with labels <= 10
#             # if len(table.get_header()) <= 10:
#             print(training_file)
#             if not os.path.exists(os.path.join(training_data_dir, os.path.dirname(training_file))):
#                 os.makedirs(os.path.join(training_data_dir, os.path.dirname(training_file)))
#             basename = training_file.rstrip('.json')
#             json.dump(table.data,
#                       open(os.path.join(training_data_dir, training_file), 'w+'),
#                       indent=4)
#             numpy.savetxt(os.path.join(training_data_dir, '{}_ner.csv'.format(basename)),
#                           table.generate_ner_matrix(st, tag_to_index), fmt='%i', delimiter=",")
#             numpy.savetxt(os.path.join(training_data_dir, '{}_wordlist.csv'.format(basename)),
#                           table.generate_wordlist_matrix(wordlist), fmt='%i', delimiter=",")


if __name__ == '__main__':
    value = '08/22/2015 - 10:00am'
    print(identify_nst('12423.5'))
