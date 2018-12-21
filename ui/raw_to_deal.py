import sys
import re
import json

tag_to_index = {'LOCATION': 0, 'PERSON': 1, 'ORGANIZATION': 2, 'MONEY': 3, 'PERCENT': 4, 'DATE': 5, 'TIME': 6}
tags = list(tag_to_index.keys())
wordlist_json = 'data/wordlist_v6.json'
wordlist = list(json.load(open(wordlist_json)).keys())


def value_to_tags(value):
    if value < 0:
        return 'NULL'
    bools = list(map(int, list('{0:b}'.format(int(value)).zfill(7))))
    return [tag for (index, tag) in enumerate(tags) if bools[6 - index]]


if __name__ == '__main__':
    filename = sys.argv[1]
    data = open(filename).read()
    feature = list(map(int, re.findall(r'-?\d+', data[:data.index('prediction')])))
    prediction = list(map(int, re.findall(r'-?\d+', data[data.index('prediction'):data.index('target')])))
    target = list(map(int, re.findall(r'-?\d+', data[data.index('target'):])))
    print('feature')
    print(list(map(value_to_tags, feature)))
    print('prediction')
    print([wordlist[index] if 0 <= index < 3333 else 'OTHER' if index == 3333 else 'NULL' for index in prediction])
    print('target')
    print([wordlist[index] if 0 <= index < 3333 else 'OTHER' if index == 3333 else 'NULL' for index in target])
