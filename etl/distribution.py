import json
from operator import itemgetter
from collections import defaultdict
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


def analyze_cut():
    with open('data/distribution_cut.txt') as f:
        lines = f.readlines()
        kvs = [line.strip('\n').split(' ') for line in lines]

    kvs_count = list(
        map(lambda kv: [kv[0], kv[1], int(kv[2][kv[2].index(':') + 1:])],
            filter(lambda kv: kv[2].startswith('count'), kvs)))
    kvs_percentage = list(
        map(lambda kv: [kv[0], kv[1], float(kv[2][kv[2].index(':') + 1:].strip('%'))],
            filter(lambda kv: kv[2].startswith('percentage'), kvs)))

    for kv in kvs_count:
        kv[0] = convert_value_to_tags(kv[0])
        kv[1] = convert_value_to_label(kv[1])

    for kv in kvs_percentage:
        kv[0] = convert_value_to_tags(kv[0])
        kv[1] = convert_value_to_label(kv[1])

    kvs_count_sorted = sorted(kvs_count, key=itemgetter(2), reverse=True)
    kvs_percentage_sorted = sorted(kvs_percentage, key=itemgetter(2), reverse=True)
    json.dump(kvs_count_sorted, open('data/report_cut_count.json', 'w+'), indent=4)
    json.dump(kvs_percentage_sorted, open('data/report_cut_percentage.json', 'w+'), indent=4)


def analyze_nocut():
    with open('data/distribution_nocut.txt') as f:
        lines = f.readlines()
        kvs = [line.strip('\n').split(' ') for line in lines]

    k_list = defaultdict(list)
    v_list = defaultdict(list)

    for kv in kvs:
        k_list[kv[0]].append(kv[1])
        v_list[kv[1]].append(kv[0])

    k_list_sorted = (
        sorted(filter(lambda item: len(item[1]) > 1, k_list.items()), key=(lambda item: len(item[1])), reverse=True))
    v_list_sorted = (sorted(filter(lambda item: len(item[1]) > 1, v_list.items()), key=(lambda item: len(item[1])), reverse=True))

    for i in range(len(k_list_sorted)):
        k_list_sorted[i] = [list(map(convert_value_to_tags, k_list_sorted[i][0].split(','))),
                            [list(map(convert_value_to_label, labels.split(','))) for labels in k_list_sorted[i][1]]]

    for i in range(len(v_list_sorted)):
        v_list_sorted[i] = [list(map(convert_value_to_label, v_list_sorted[i][0].split(','))),
                            [list(map(convert_value_to_tags, labels.split(','))) for labels in v_list_sorted[i][1]]]
    json.dump(k_list_sorted, open('data/report_nocut_1int.json', 'w+'), indent=4)
    json.dump(v_list_sorted, open('data/report_nocut_ni1t.json', 'w+'), indent=4)

    # print(v_list_sorted)


if __name__ == '__main__':
    analyze_cut()
    analyze_nocut()
