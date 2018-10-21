import json
from .tagger import tag_to_index

wordlist = json.load(open('data/wordlist.json'))


def convert_value_to_tags(value):
    if int(value) < 0:
        return []
    else:
        bin_str = '{0:b}'.format(int(value)).zfill(7)
        bool_list = [digit == '1' for digit in bin_str]
        return list(filter(lambda tag: bool_list[6 - tag_to_index[tag]], tag_to_index))


def convert_value_to_label(value):
    if int(value) >= 4510 or int(value) < 0:
        return 'OTHER'
    else:
        return list(wordlist.keys())[int(value)]


if __name__ == '__main__':
    with open('data/distribution_nocut.txt') as f:
        lines = f.readlines()
        kvs = [line.strip('\n').split(' ') for line in lines]

    for kv in kvs:
        kv[0] = list(map(convert_value_to_tags, kv[0].split(',')))
        kv[1] = list(map(convert_value_to_label, kv[1].split(',')))

    print(kvs)
