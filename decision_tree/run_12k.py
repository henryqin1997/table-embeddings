import json
from collections import defaultdict

import numpy as np
from .decision_tree import diction_pred_advanced
from .decision_tree import diction_pred
from .load import load_data_12k,load_data_12k_with_raw
from urllib.parse import urlparse

def measure_distribution_cut(diction, input, target):
    for i in range(len(input)):
        if input[i]!=-1:
            diction[str(input[i])][str(target[i])]+=1
        else:
            break


def measure_distribution_no_cut(diction, input, target):
    input = [str(x) for x in input]
    target = [str(x) for x in target]
    diction[','.join(input)][','.join(target)] += 1

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
            #print(input[i])
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
        input, target = load_data_12k(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1
        for i in range(len(input)):

            total += 1
            pred = diction_pred(dic_prediction,dic_cut_pred,input[i])

            for j in range(len(target[i])):
                if target[i][j]!=-1:
                    total+=1
                    if pred[j]==target[i][j]:
                        correct+=1
                else:
                    break
    print('validation accuracy {}'.format(correct / total))

def test():

    dic_pred=json.load(open('decision_tree/diction_prediction_advanced.json'))
    #dic_cut_pred=json.load(open('decision_tree/dic_12k_cut_pred.json'))

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

            pred = diction_pred_advanced(dic_pred, input[i])
            for j in range(len(target[i])):
                if target[i][j] != -1:
                    total += 1
                    if pred[j] == target[i][j]:
                        correct += 1
                else:
                    break
    print('validation accuracy {}'.format(correct / total))

def accuracy_12k_no_other():
    diction = json.load(open('decision_tree/diction_12k.json'))
    correct = 0
    sum = 0

    for key in diction.keys():
        max = 0
        for subkey in diction[key]:
            if diction[key][subkey]>max:
                max=diction[key][subkey]
                maxlabel=subkey
        pred = [int(x) for x in maxlabel.split(',') if int(x)!=-1]
        for subkey in diction[key]:
            case = [int(x) for x in subkey.split(',') if int(x)!=-1]
            for i in range(len(pred)):
                if case[i]!=3333:
                    sum+=diction[key][subkey]
                    if pred[i]==case[i]:
                        correct+=diction[key][subkey]
    print('accuracy no other" {}'.format(correct/sum))

def rank_cl_pl_pairs():
    '''Rank (cc,pc) pairs with their count and save to diction. Also want raw data for incorrect predictions.'''
    dic_cut_pred = json.load(open('decision_tree/dic_cut_pred.json', 'r'))
    dic_pred = json.load(open('decision_tree/diction_prediction_with0.json', 'r'))
    #dic_pred_ad=json.load(open('decision_tree/diction_prediction_advanced.json', 'r'))
    #cl_pl_count = defaultdict(int)
    cl_pl_qualified = defaultdict(lambda: [])

    train_size = 100000
    batch_size = 50
    batch_index = 0

    while batch_size * batch_index < train_size:
        print(batch_index)
        raw, input, target = load_data_12k_with_raw(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1

        for j in range(len(input)):

            feature = input[j]

            parsed_uri = urlparse(raw[j]["url"])
            result = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)


            if ','.join(str(x) for x in feature) not in dic_pred:
                pred = diction_pred(dic_pred,dic_cut_pred,feature)
                #pred = diction_pred_advanced(dic_pred_ad,feature)
                for i in range(10):

                    if target[j][i] != -1:
                        #cl_pl_count[(target[j][i], pred[i])] += 1
                        if result not in cl_pl_qualified[(target[j][i], pred[i])]:
                            cl_pl_qualified[(target[j][i], pred[i])].append(result)
                    else:
                        break

            else:
                pred = dic_pred[','.join(str(x) for x in feature)]
                #pred = dic_pred_ad[','.join(str(x) for x in feature)][0]
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

    with open('decision_tree/clpl_qualified_version4_count.json', 'w') as wfp:
        json.dump(dic_new, wfp)



if __name__=='__main__':
    #train()
    #test()
    accuracy_12k_no_other()
