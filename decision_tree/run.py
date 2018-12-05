import json
from collections import defaultdict

import numpy as np
import decision_tree
from .decision_tree import diction_pred
from .load import load_data, load_data_with_raw, load_data_100_sample_with_raw
from urllib.parse import urlparse

def judge_qualified(key_transformed,col):
    if col>10 or col<0:
        print('wrong col {} for qualified!'.format(col))
        exit(0)
    qualified = True
    for i in range(col):
        qualified = qualified and key_transformed[i] != 0 and key_transformed[i] != -1
    return qualified

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


def train():
    dic_cut = defaultdict(lambda: defaultdict(int))
    dic_no_cut = defaultdict(lambda: defaultdict(int))
    dic_prediction = defaultdict(lambda: '')

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

    with open('decision_tree/diction.json', 'w') as fp:
        json.dump(dic_no_cut, fp)
        print('diction saved')

    with open('decision_tree/diction_cut.json', 'w') as fp:
        json.dump(dic_cut, fp)
        print('diction_cut saved')

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
                sum += dic_no_cut[key][label]
                if dic_no_cut[key][label] > max:
                    max = dic_no_cut[key][label]
                    maxlabel = label
                # print(key, label, 'count:{}'.format(dic_no_cut[key][label]))
        pre_acc += max
        dic_prediction[key] = maxlabel
    print("train accuracy {}".format(pre_acc / sum))

    with open('decision_tree/diction_prediction_with0.json', 'w') as fp:
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

    with open('decision_tree/dic_cut_pred.json', 'w') as fp:
        json.dump(dic_cut_pred, fp)

    print('validating')
    batch_size = 50
    batch_index = 2000
    correct = 0
    total = 0
    while batch_size * batch_index < 103000:
        print(batch_index)
        input, target = load_data(batch_size=batch_size, batch_index=batch_index)
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
            key = ','.join(key_list)
            value = ','.join(value_list)
            if dic_prediction[key] == value:
                correct += 1
    print('validation accuracy {}'.format(correct / total))


def rank_cl_pl_pairs():
    '''Rank (cc,pc) pairs with their count and save to diction. Also want raw data for incorrect predictions.'''
    dic_cut_pred = json.load(open('decision_tree/dic_cut_pred.json', 'r'))
    dic_pred = json.load(open('decision_tree/diction_prediction_with0.json', 'r'))

    #cl_pl_count = defaultdict(int)
    cl_pl_qualified = defaultdict(lambda: [])

    train_size = 100000
    batch_size = 50
    batch_index = 0

    while batch_size * batch_index < train_size:
        print(batch_index)
        raw, input, target = load_data_with_raw(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1

        for j in range(len(input)):

            feature = input[j]

            parsed_uri = urlparse(raw[j]["url"])
            result = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

            if not judge_qualified(feature,5):
                continue

            if ','.join(str(x) for x in feature) not in dic_pred:
                pred = diction_pred(dic_pred,dic_cut_pred,feature)
                for i in range(10):

                    if target[j][i] != -1:
                        #cl_pl_count[(target[j][i], pred[i])] += 1
                        if result not in cl_pl_qualified[(target[j][i], pred[i])]:
                            cl_pl_qualified[(target[j][i], pred[i])].append(result)
                    else:
                        break

            else:
                pred = dic_pred[','.join(str(x) for x in feature)]
                pred = [int(x) for x in pred.split(',')]
                for i in range(10):
                    if target[j][i] != -1:
                        #cl_pl_count[(target[j][i], pred[i])] += 1
                        if result not in cl_pl_qualified[(target[j][i], pred[i])]:
                            cl_pl_qualified[(target[j][i], pred[i])].append(result)
                    else:
                        break

    dic_new = {}
    # for key in cl_pl_count.keys():
    #     dic_new[str(key)] = cl_pl_count[key]

    for key in cl_pl_qualified.keys():
        dic_new[str(key)] = len(cl_pl_qualified[key])

    with open('decision_tree/clpl_qualified_count.json', 'w') as wfp:
        json.dump(dic_new, wfp)

    # acc = 0
    # sum = 0
    # for key in cl_pl_count.keys():
    #     sum += cl_pl_count[key]
    #     if key[0] == key[1]:
    #         acc += cl_pl_count[key]
    # print('label accuracy is {}'.format(float(acc / sum)))


def test():
    return 0


def calculate_label_accuracy():
    with open('decision_tree/clpl_count.json', 'r') as rfp:
        cl_pl_count = json.load(rfp)
    acc = 0
    sum = 0
    for key in cl_pl_count.keys():
        sum += cl_pl_count[key]
        key_transformed = eval(key)
        print(key_transformed)
        if key_transformed[0] == key_transformed[1]:
            acc += cl_pl_count[key]

    print('label accuracy is {}'.format(float(acc / sum)))


def draw_raw():
    with open('decision_tree/clpl_qualified_count.json', 'r') as rfp:
        cl_pl_count = json.load(rfp)
    maxlist = [['', 0] for i in range(5)]
    for key in cl_pl_count.keys():
        key_transformed = eval(key)
        if key_transformed[0] != key_transformed[1] and key_transformed[0] !=3333 and key_transformed[1] !=3333:
            max_counts = np.array([maxlist[j][1] for j in range(5)])
            i = np.random.choice(np.flatnonzero(max_counts == max_counts.min()))
            if cl_pl_count[key] > maxlist[i][1]:
                maxlist[i][1] = cl_pl_count[key]
                maxlist[i][0] = key_transformed
    print(maxlist)

    raw_check = [x[0] for x in maxlist]

    dic_cut_pred = json.load(open('decision_tree/dic_cut_pred.json', 'r'))
    dic_pred = json.load(open('decision_tree/diction_prediction_with0.json', 'r'))

    train_size = 100000
    batch_size = 50
    batch_index = 0

    while batch_size * batch_index < train_size:
        print(batch_index)
        raw, input, target = load_data_with_raw(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1

        for j in range(len(input)):

            feature = input[j]
            if ','.join(str(x) for x in feature) not in dic_pred:
                pred = diction_pred(dic_pred,dic_cut_pred,feature)
                for i in range(10):
                    if target[j][i] != -1:
                        new_key = str((target[j][i], pred[i]))
                        if (target[j][i], pred[i]) in raw_check:
                            with open('raw_to_deal_5_no_other/{}_{}.json'.format(new_key, cl_pl_count[
                                new_key]), 'w') as wfp:
                                json.dump(raw[j], wfp)

                            with open('raw_to_deal_5_no_other/{}_{}.txt'.format(new_key,
                                                                     cl_pl_count[new_key]), 'w') as wfp:
                                wfp.write('feature')
                                wfp.write(str(feature))
                                wfp.write('prediction')
                                wfp.write(str(pred))
                                wfp.write('target')
                                wfp.write(str(target[j]))
                                cl_pl_count[str((target[j][i], pred[i]))] -= 1
                                break
                    else:
                        break

            else:
                pred = dic_pred[','.join(str(x) for x in feature)]
                pred = [int(x) for x in pred.split(',')]
                for i in range(10):
                    if target[j][i] != -1:
                        new_key = str((target[j][i], pred[i]))
                        if (target[j][i], pred[i]) in raw_check:
                            with open('raw_to_deal_5_no_other/{}_{}.json'.format(new_key, cl_pl_count[
                                new_key]), 'w') as wfp:
                                json.dump(raw[j], wfp)
                            with open('raw_to_deal_5_no_other/{}_{}.txt'.format(new_key,
                                                                     cl_pl_count[new_key]), 'w') as wfp:
                                wfp.write('feature')
                                wfp.write(str(feature))
                                wfp.write('prediction')
                                wfp.write(str(pred))
                                wfp.write('target')
                                wfp.write(str(target[j]))
                                cl_pl_count[str((target[j][i], pred[i]))] -= 1
                                break
                    else:
                        break

def filter_feature(dic):
    diction={}
    for key in dic.keys():
        key_transformed = [int(x) for x in key.split(',')]
        qualified = True
        for i in range(5):
            qualified = qualified and key_transformed[i]!=0 and key_transformed[i]!=-1
        if qualified:
            diction[key]=dic[key]
    with open('decision_tree/qualified_prediction.json','w') as fp:
        json.dump(diction,fp)

def generate_dic_pred():
    prediction=defaultdict(lambda: [])
    with open('decision_tree/diction.json','r') as fp:
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
    with open('decision_tree/diction_prediction_advanced.json','w') as fp:
        json.dump(prediction,fp)


if __name__ == '__main__':
    # train()
    #rank_cl_pl_pairs()
    draw_raw()
    #generate_dic_pred()
