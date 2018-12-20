import json
from collections import defaultdict

import numpy as np
from .decision_tree import diction_pred_advanced
from .decision_tree import diction_pred
from .load import load_data_12k
from urllib.parse import urlparse

def measure_distribution_cut(diction, input, target):
    input_transformed = input.transpose()
    target_transformed = target.transpose()
    for index, row in enumerate(input_transformed):
        if row[0] == 0:
            try:
                i = list(row).index(1)
                t = list(target_transformed[index]).index(1)
                diction[i][t] += 1
            except ValueError:
                pass


def measure_distribution_no_cut(diction, input, target):
    input_transformed = input.transpose()
    target_transformed = target.transpose()
    key_list = []
    value_list = []
    for index, row in enumerate(input_transformed):
        try:
            i = list(row).index(1)
            t = list(target_transformed[index]).index(1)
        except ValueError:
            i = -1
            t = -1
        finally:
            key_list.append(str(i))
            value_list.append(str(t))
    diction[','.join(key_list)][','.join(value_list)] += 1

def generate_dic_pred():
    prediction=defaultdict(lambda: [])
    with open('decision_tree/diction_12k.json','r') as fp:
        dic = json.load(fp)
    for key in dic.keys():
        length=len([x for x in key.split(',') if x!='-1'])
        findmax=[defaultdict(int) for i in range(length)]
        for seckey in dic[key].keys():
            seckey_transformed=[x for x in seckey.split(',')]
            for i in range(length):
                findmax[i][seckey_transformed[i]]+=dic[key][seckey]
        maxlabel = ['-1' for i in range(10)]
        maxcount = [0 for i in range(10)]
        for i in range(10):
            if i<length:
                total=0
                for label,count in findmax[i].items():
                    total+=count
                    if count>maxcount[i]:
                        maxcount[i]=count
                        maxlabel[i]=label
                maxcount[i]=float(maxcount[i]/total)
            else:
                maxcount[i]=1.0
        prediction[key].append(','.join(maxlabel))
        prediction[key].append(maxcount)
    with open('decision_tree/diction_12k_prediction_advanced.json','w') as fp:
        json.dump(prediction,fp)


def train():
    dic_cut = defaultdict(lambda: defaultdict(int))
    dic_no_cut = defaultdict(lambda: defaultdict(int))
    dic_prediction = defaultdict(lambda: '')

    train_size = 12000
    batch_size = 50
    batch_index = 0
    while batch_size * batch_index < train_size:
        print(batch_index)
        input, target = load_data_12k(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1
        for i in range(len(input)):
            measure_distribution_cut(dic_cut, input[i], target[i])
            measure_distribution_no_cut(dic_no_cut, input[i], target[i])

    with open('decision_tree/diction_12k.json', 'w') as fp:
        json.dump(dic_no_cut, fp)
        print('diction saved')

    with open('decision_tree/diction_12k_cut.json', 'w') as fp:
        json.dump(dic_cut, fp)
        print('diction_12k_cut saved')

    print('table')

    # with open('diction_12k.json', 'r') as fp:
    #     dic_no_cut = json.load(fp)

    #pre_acc = 0
    #sum = 0

    for key in dic_no_cut.keys():
        max = 0
        maxlabel = ''
        for label in dic_no_cut[key].keys():
            if label != '-1,-1,-1,-1,-1,-1,-1,-1,-1,-1':
                #sum += dic_no_cut[key][label]
                if dic_no_cut[key][label] > max:
                    max = dic_no_cut[key][label]
                    maxlabel = label
                # print(key, label, 'count:{}'.format(dic_no_cut[key][label]))
        #pre_acc += max
        dic_prediction[key] = maxlabel
    #print("train accuracy {}".format(pre_acc / sum))

    with open('decision_tree/diction_12k_prediction_with0.json', 'w') as fp:
        json.dump(dic_prediction, fp)
        print('diciton prediction saved')

    generate_dic_pred()

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

    with open('decision_tree/dic_12k_cut_pred.json', 'w') as fp:
        json.dump(dic_cut_pred, fp)

    print('validating')
    batch_size = 50
    batch_index = 240
    correct = 0
    total = 0
    while batch_size * batch_index < 12200:
        print(batch_index)
        input, target = load_data_12k(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1
        for i in range(len(input)):
            total += 1
            input_transformed = input[i].transpose()
            target_transformed = target[i].transpose()
            key_list = []
            value_list = []
            for index, row in enumerate(input_transformed):
                try:
                    i = list(row).index(1)
                    t = list(target_transformed[index]).index(1)
                except ValueError:
                    i = -1
                    t = -1
                finally:
                    key_list.append(str(i))
                    value_list.append(str(t))

            pred = diction_pred(dic_prediction,dic_cut_pred,key_list)
            for i in range(len(value_list)):
                if value_list[i]!=-1:
                    total+=1
                    if pred[i]==value_list[i]:
                        correct+=1
                else:
                    break
    print('validation accuracy {}'.format(correct / total))

def test():

    dic_pred=json.load('decision_tree/diction_12k_prediction_with0.json', 'r')
    dic_cut_pred=json.load('decision_tree/dic_12k_cut_pred.json','r')

    print('overall validating')
    batch_size = 50
    batch_index = 0
    correct = 0
    total = 0
    while batch_size * batch_index < 12200:
        print(batch_index)
        input, target = load_data_12k(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1
        for i in range(len(input)):
            total += 1
            input_transformed = input[i].transpose()
            target_transformed = target[i].transpose()
            key_list = []
            value_list = []
            for index, row in enumerate(input_transformed):
                try:
                    i = list(row).index(1)
                    t = list(target_transformed[index]).index(1)
                except ValueError:
                    i = -1
                    t = -1
                finally:
                    key_list.append(str(i))
                    value_list.append(str(t))

            pred = diction_pred(dic_pred, dic_cut_pred, key_list)
            for i in range(len(value_list)):
                if value_list[i] != -1:
                    total += 1
                    if pred[i] == value_list[i]:
                        correct += 1
                else:
                    break
    print('validation accuracy {}'.format(correct / total))


if __name__=='__main__':
    train()
    test()

