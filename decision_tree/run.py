import json
from collections import defaultdict

import numpy as np
from .decision_tree import diction_pred_advanced
from .decision_tree import diction_pred
from .decision_tree import label_num_arr, label_num_str, label_num_str_no_other, correct_pred, correct_pred_no_other
from .load import load_data
from .load import load_data_100_sample


def measure_distribution_cut(diction, input, target):
    for i in range(len(input)):
        if input[i] != -1:
            diction[str(input[i])][str(target[i])] += 1
        else:
            break


def measure_distribution_no_cut(diction, input, target):
    input = [str(x) for x in input]
    target = [str(x) for x in target]
    diction[','.join(input)][','.join(target)] += 1


def generate_dic_pred():
    prediction = defaultdict(lambda: [])
    with open('decision_tree/diction.json', 'r') as fp:
        dic = json.load(fp)
    for key in dic.keys():
        length = len([x for x in key.split(',') if x != '-1'])
        findmax = [defaultdict(int) for i in range(length)]
        for seckey in dic[key].keys():
            seckey_transformed = [x for x in seckey.split(',')]
            for i in range(length):
                findmax[i][seckey_transformed[i]] += dic[key][seckey]
        maxlabel = ['-1' for i in range(10)]
        maxcount = [0 for i in range(10)]
        for i in range(10):
            if i < length:
                total = 0
                for label, count in findmax[i].items():
                    total += count
                    if count > maxcount[i]:
                        maxcount[i] = count
                        maxlabel[i] = label
                maxcount[i] = float(maxcount[i] / total)
            else:
                maxcount[i] = 1.0
        prediction[key].append(','.join(maxlabel))
        prediction[key].append(maxcount)
    with open('decision_tree/diction_prediction_advanced.json', 'w') as fp:
        json.dump(prediction, fp, indent=4)


def train():
    dic_cut = defaultdict(lambda: defaultdict(int))
    dic_no_cut = defaultdict(lambda: defaultdict(int))
    dic_prediction = defaultdict(lambda: '')

    # for "10digits, load_data in enumerate(
    #         [load_nst_major, load_nst_max, load_nst_overall, load_nst_mm, load_nst_majo, load_nst_maxo, load_nst_mmo]):

    train_size = 100000
    batch_size = 50
    batch_index = 0
    while batch_size * batch_index < train_size:
        print(batch_index)
        input, target = load_data(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1
        for i in range(len(input)):
            measure_distribution_cut(dic_cut, input[i], target[i])
            measure_distribution_no_cut(dic_no_cut, input[i], target[i])

    with open('decision_tree/diction_nst{}.json'.format("10digits"), 'w') as fp:
        json.dump(dic_no_cut, fp, indent=4)
        print('diction_nst{} saved'.format("10digits"))

    with open('decision_tree/diction_nst_cut{}.json'.format("10digits"), 'w') as fp:
        json.dump(dic_cut, fp, indent=4)
        print('diction_cut{} saved'.format("10digits"))

    print('table')

    # with open('diction.json', 'r') as fp:
    #     dic_no_cut = json.load(fp)

    pre_acc = 0
    sum = 0

    for key in dic_no_cut.keys():
        max = 0
        maxlabel = ''
        for label in dic_no_cut[key].keys():
            if label != '-1,-1,-1,-1,-1,-1,-1,-1,-1,-1':
                sum += dic_no_cut[key][label] * label_num_str(label)
                if dic_no_cut[key][label] > max:
                    max = dic_no_cut[key][label]
                    maxlabel = label
                # print(key, label, 'count:{}'.format(dic_no_cut[key][label]))
            else:
                continue

        dic_prediction[key] = maxlabel
        for label in dic_no_cut[key].keys():
            pre_acc += dic_no_cut[key][label] * correct_pred(maxlabel, label)

    print("train accuracy {}".format(pre_acc / sum))

    with open('decision_tree/diction_nst_prediction{}.json'.format("10digits"), 'w') as fp:
        json.dump(dic_prediction, fp, indent=4)
        print('diciton prediction{} saved'.format("10digits"))
    # generate_dic_pred()

    dic_cut_pred = defaultdict(lambda: ['', 0.])
    for key1 in dic_cut.keys():
        sum_num = 0
        max = 0
        maxlabel = ''
        for key in dic_cut[key1].keys():
            sum_num += dic_cut[key1][key]
            if dic_cut[key1][key] > max:
                max = dic_cut[key1][key]
                maxlabel = key
        dic_cut_pred[key1] = [maxlabel, float(max / sum_num)]

    with open('decision_tree/dic_nst_cut_pred{}.json'.format("10digits"), 'w') as fp:
        json.dump(dic_cut_pred, fp, indent=4)

    print('validating')
    batch_size = 50
    batch_index = 2000
    correct = 0
    total = 0
    total_noOther = 0
    correct_noOther = 0
    dic_result = defaultdict(lambda: defaultdict(int))
    while batch_size * batch_index < 103000:
        print(batch_index)
        input, target = load_data(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1
        for i in range(len(input)):

            total += 1
            pred = diction_pred(dic_prediction, dic_cut_pred, input[i])

            dic_result[','.join(target[i])][','.join(pred)] += 1

            for j in range(len(target[i])):
                if target[i][j] != -1:
                    total += 1
                    if pred[j] == target[i][j]:
                        correct += 1
                    if target[i][j] != 3333:
                        total_noOther += 1
                        if pred[j] == target[i][j]:
                            correct_noOther += 1
                else:
                    break
    with open('decision_tree/dic_result_10digit.json', 'w')as fp:
        json.dump(sort_by_error_frequency(dic_result), fp, indent=4)
    print('validation {} accuracy {}'.format("10digits", correct / total))
    print('no other validation {} accuracy {}'.format("10digits", correct_noOther / total_noOther))


def cal_accuracy():
    dic_no_cut = json.load(open('decision_tree/diction_nst{}.json'.format("10digits")))
    pre_acc = 0
    sum = 0
    sum_no_other = 0
    acc_no_other = 0

    for key in dic_no_cut.keys():
        max = 0
        maxlabel = ''
        for label in dic_no_cut[key].keys():
            if label != '-1,-1,-1,-1,-1,-1,-1,-1,-1,-1':
                sum += dic_no_cut[key][label] * label_num_str(label)
                sum_no_other += dic_no_cut[key][label] * label_num_str_no_other(label)
                if dic_no_cut[key][label] > max:
                    max = dic_no_cut[key][label]
                    maxlabel = label
                # print(key, label, 'count:{}'.format(dic_no_cut[key][label]))
            else:
                continue
        for label in dic_no_cut[key].keys():
            pre_acc += dic_no_cut[key][label] * correct_pred(maxlabel, label)
            acc_no_other += dic_no_cut[key][label] * correct_pred_no_other(maxlabel, label)

    print("train accuracy {}".format(pre_acc / sum))
    print("train accuracy no other {}".format(acc_no_other / sum_no_other))

    dic_prediction = json.load(open('decision_tree/diction_nst_prediction{}.json'.format("10digits")))
    dic_cut_pred = json.load(open('decision_tree/dic_nst_cut_pred{}.json'.format("10digits")))

    batch_size = 50
    batch_index = 2000
    correct = 0
    total = 0
    total_noOther = 0
    correct_noOther = 0
    dic_result = defaultdict(lambda: defaultdict(int))
    while batch_size * batch_index < 103000:
        print(batch_index)
        input, target = load_data(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1
        for i in range(len(input)):

            total += 1
            pred = diction_pred(dic_prediction, dic_cut_pred, input[i])

            target_str = ','.join([str(x) for x in target[i]])
            pred_str = ','.join([str(x) for x in pred])
            dic_result[target_str][pred_str] += 1

            for j in range(len(target[i])):
                if target[i][j] != -1:
                    total += 1
                    if pred[j] == target[i][j]:
                        correct += 1
                    if target[i][j] != 3333:
                        total_noOther += 1
                        if pred[j] == target[i][j]:
                            correct_noOther += 1
                else:
                    break
    with open('decision_tree/dic_result_10digit.json', 'w')as fp:
        json.dump(sort_by_error_frequency(dic_result), fp, indent=4)
    print('validation {} accuracy {}'.format("10digits", correct / total))
    print('no other validation {} accuracy {}'.format("10digits", correct_noOther / total_noOther))


def error_frequency(item):
    key, value = item
    sum = 0
    for pred in value:
        if pred != key:
            sum += value[pred]
    return sum


def sort_by_error_frequency(dic_result):
    return dict(sorted(dic_result.items(), key=error_frequency, reverse=True))


if __name__ == '__main__':
    # train()
    cal_accuracy()
