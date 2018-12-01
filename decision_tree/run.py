import json
from collections import defaultdict
from .load import load_data,load_data_with_raw,load_sample_random_label,load_sample_random_table
from .decision_tree import diction_pred

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


    dic_cut_pred=defaultdict(lambda: ['',0.])
    for key1 in dic_cut.keys():
        sum_num=0
        max=0
        maxlabel=''
        for key in dic_cut[key1].keys():
            sum_num+=dic_cut[key1][key]
            if dic_cut[key1][key]>max:
                max=dic_cut[key1][key]
                maxlabel=key
        dic_cut_pred[key1]=[maxlabel,float(max/sum_num)]

    with open('decision_tree/dic_cut_pred.json','w') as fp:
        json.dump(dic_cut_pred,fp)


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

def rank_cc_pc_pairs():
    '''Rank (cc,pc) pairs with their count and save to diction. Also want raw data for incorrect predictions.'''
    dic_cut_pred = json.load(open('decision_tree/dic_cut_pred.json','r'))
    dic_pred = json.load(open('decision_tree/diction_prediction_with0.json', 'r'))

    cc_pc_count = defaultdict(int)

    train_size = 100000
    batch_size = 50
    batch_index = 0

    while batch_size * batch_index < train_size:
        print(batch_index)
        input, target = load_data(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1

        for j in range(len(input)):

            feature = input[j]
            if ','.join(str(x) for x in feature) not in dic_pred:
                sel_feature, pred = diction_pred(dic_pred, feature)
                pred = [int(x) for x in pred.split(',')]
                for i in range(10):

                    if target[j][i] != -1:

                        if feature[i] != sel_feature[i] and str(feature[i]) in dic_cut_pred.keys():
                            if dic_cut_pred[str(feature[i])][1] >= 0.5:
                                cc_pc_count[(target[j][i], dic_cut_pred[str(feature[i])][0])] += 1
                            else:
                                cc_pc_count[(target[j][i], pred[i])] += 1

                        else:
                            cc_pc_count[(target[j][i], pred[i])] += 1
                    else:
                        break

            else:
                pred = dic_pred[','.join(str(x) for x in feature)]
                pred = [int(x) for x in pred.split(',')]
                for i in range(10):
                    if target[j][i] != -1:
                        cc_pc_count[(target[j][i],pred[i])]+=1
                    else:
                        break

    dic_new={}
    for key in cc_pc_count.keys():
        dic_new[str(key)]=cc_pc_count[key]

    with open('ccpc_count.json','w') as wfp:
        json.dump(dic_new,wfp)

    acc=0
    sum=0
    for key in cc_pc_count.keys():
        sum+=cc_pc_count[key]
        if key[0]==key[1]:
            acc+=1
    print('label accuracy is {}'.format(float(acc/sum)))

    return 0

def test():
    return 0


if __name__=='__main__':
    #train()
    rank_cc_pc_pairs()
