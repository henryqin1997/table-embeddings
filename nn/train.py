import json
import os
import numpy
from collections import defaultdict
from etl import Table

training_data_dir = 'data/train'
training_files_json = 'data/training_files_filtered.json'
training_files = json.load(open(training_files_json))
testing_data_dir = 'data/sample_random_label_test'
activate_data_dir = 'data/sample_random_label'
testing_files_json = 'data/testing_files_random_label.json'
# testing_files_json = 'data/testing_files_random_label_sample.json'
testing_files = [[y for y in json.load(open(testing_files_json)) if y[0] == str(x)] for x in range(10)]
tag_to_index = {'LOCATION': 0, 'PERSON': 1, 'ORGANIZATION': 2, 'MONEY': 3, 'PERCENT': 4, 'DATE': 5, 'TIME': 6}


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)

def one_hot(row):
    assert len(row) > 0
    row_sum = int(round(sum(numpy.array([(2 ** i) * num for (i, num) in enumerate(row)]))))
    row_converted = numpy.zeros(2 ** len(row))
    assert row_sum < len(row_converted)
    row_converted[row_sum] = 1
    return row_converted


def load_data(batch_size, batch_index=0):
    # load training data from file, to be implemented
    # put size number of data into one array
    # start from batch_index batch
    batch_files = training_files[batch_size * batch_index:batch_size * (batch_index + 1)]
    batch_files_ner = list(map(lambda batch_file: batch_file.rstrip('.json') + '_ner.csv', batch_files))
    batch_files_wordlist = list(map(lambda batch_file: batch_file.rstrip('.json') + '_wordlist.csv', batch_files))
    inputs = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_ner), delimiter=',') for batch_file_ner in
         batch_files_ner])
    targets = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_wordlist), delimiter=',') for batch_file_wordlist
         in batch_files_wordlist])

    inputs_transformed = []
    targets_transformed = []

    # Use One Hot Encoding and remove column with all zeros
    for i in range(len(inputs)):
        input = inputs[i]
        target = targets[i]
        assert len(input) == len(tag_to_index)

        input = numpy.array([one_hot(row) for row in input.transpose()])
        target = target.transpose()

        # Remove all zero columns
        input_transformed = numpy.zeros(input.shape)
        target_transformed = numpy.zeros(target.shape)

        current_index = 0
        for row in input:
            if row[0] == 0:
                input_transformed[current_index] = input[current_index]
                target_transformed[current_index] = target[current_index]
                current_index += 1

        inputs_transformed.append(input_transformed.transpose())
        targets_transformed.append(target_transformed.transpose())
    return numpy.array(inputs_transformed), numpy.array(targets_transformed)


def indexOf(l, n):
    try:
        return list(l).index(n)
    except ValueError:
        return -1


def load_sample_random_label(sample_index, batch_size, batch_index):
    # load testing data of sample with random labels
    # put size number of data into one array
    # start from batch_index batch
    result = []
    batch_files = testing_files[sample_index][batch_size * batch_index:batch_size * (batch_index + 1)]
    for batch_file in batch_files:
        table = Table(json.load(open(os.path.join(testing_data_dir, batch_file))))
        column_num = len(table.get_header())
        batch_file_ner = batch_file.rstrip('.json') + '_ner.csv'
        batch_file_wordlist = batch_file.rstrip('.json') + '_wordlist.csv'
        batch_file_activate = batch_file.rstrip('.json') + '_activate.json'
        input = numpy.genfromtxt(os.path.join(testing_data_dir, batch_file_ner), delimiter=',').transpose()
        target = numpy.genfromtxt(os.path.join(testing_data_dir, batch_file_wordlist), delimiter=',').transpose()
        activate = json.load(open(os.path.join(activate_data_dir, batch_file_activate)))

        input_transformed = [
            int(round(sum(numpy.array([(2 ** i) * num for (i, num) in enumerate(row)]))))
            if idx < column_num else -1 for idx, row in enumerate(input)]
        target_transformed = [indexOf(list(map(lambda num: int(round(num)), row)), 1) if idx < column_num else -1 for
                              idx, row in enumerate(target)]
        activate_transformed = [num if idx < column_num else -1 for idx, num in enumerate(activate)]

        result.append([input_transformed, target_transformed, activate_transformed])
    return result

def sample_dict(sample_data,sample_summary,missed_feature,faultdic,prediction):
    batch_size=len(sample_data)
    #missed_feature=[]
    #sample_summary = defaultdict(lambda: [0, 0])
    # for batch in range(iteration):
    #     sample_summary=defaultdict(lambda:[0,0])
    #     features=sample_feature[batch*batch_size:batch_size*(batch+1)]
    #     targets=sample_target[batch*batch_size:batch_size*(batch+1)]
    #     actives=sample_active[batch*batch_size:batch_size*(batch+1)]

    for index in range(batch_size):
        feature=sample_data[index][0]
        target=sample_data[index][1]
        activate=sample_data[index][2]
        if ','.join(str(x) for x in feature) not in prediction:
            pred = diction_pred(prediction,feature)
            for i in range(10):
                if activate[i]==1 and target[i]!=-1:
                    sample_summary[target[i]][1]+=1
                    if int(pred.split(',')[i])==target[i]:
                        sample_summary[target[i]][0] += 1
            missed_feature.add(','.join(str(x) for x in feature))
        else:
            for i in range(10):
                if activate[i]==1 and target[i]!=-1:
                    sample_summary[target[i]][1]+=1
                    if target[i]==int(prediction[','.join(str(x) for x in feature)].split(',')[i]):
                        sample_summary[target[i]][0]+=1
                    else:
                        mispred = [str(x) for x in target]
                        for i in range(10):
                            if activate[i]==1:
                                mispred[i]+='*'
                        mispred=','.join(mispred)
                        faultdic[','.join(str(x) for x in feature)].append(mispred)
    # with open("sample_dict_batch={}".format(batch_index),'w') as wfp:
    #         json.dump(sample_summary,wfp)
    #
    # with open("miss_features_batch={}".format(batch_index),'w') as wfp:
    #     for f in missed_feature:
    #         wfp.write(f+"\n")

def diction_pred(dic,feature):
    maxkey=''
    maxjac=0
    feature_processed=[x for x in feature if x!=-1]
    for key in dic.keys():
        key_processed = [int(x) for x in key.split(',') if x!='-1']
        jac = jaccard_similarity(feature_processed,key_processed)
        if jac>maxjac:
            maxkey=key
            maxjac=jac
    return dic[maxkey]

def sample_print():
    batch_size=50
    sample_size=4500
    missed_feature=set([])
    faultdic = defaultdict(lambda: [])
    with open('nn/diction_prediction_with0.json', 'r') as fp:
        prediction = json.load(fp)
    for sample_index in range(10):
        sample_summary = defaultdict(lambda: [0, 0])
        batch_index = 0
        while batch_size*batch_index<sample_size:
            sample_data=load_sample_random_label(sample_index,batch_size,batch_index)
            sample_dict(sample_data,sample_summary,missed_feature,faultdic,prediction)
            batch_index+=1
        with open("nn/sample_dict_it={}".format(sample_index), 'w') as wfp:
            json.dump(sample_summary, wfp)
    with open("nn/miss_features",'w') as wfp:
        for f in missed_feature:
            wfp.write(f+"\n")
    with open("nn/fault_diction.json",'w') as wfp:
        json.dump(faultdic,wfp)




# def sample_print_uniform():
#     batch_size = 50
#     sample_size = 4500
#     missed_feature = set([])
#     sample_summary = defaultdict(lambda: [0, 0])
#     batch_index = 0
#     while batch_size * batch_index < sample_size:
#         sample_data = load_sample_random_label(sample_index, batch_size, batch_index)
#         sample_dict(sample_data, sample_summary, missed_feature)
#         batch_index += 1
#     with open("nn/sample_dict_it={}".format(sample_index), 'w') as wfp:
#         json.dump(sample_summary, wfp)
#     with open("nn/miss_features", 'w') as wfp:
#         for f in missed_feature:
#             wfp.write(f + "\n")

##########################333#3#
# evaluation of model:
# 1. accuracy of prediction of label over target     #correct prediction/#targetlabel
# 2. accuracy of prediction of label over target when other doesn't count
#   #correct prediction(no 'other')/#targetlabel(no other)

def accuracy(prediction, target, batch_size=1):  # to be implemented
    if batch_size > 1:
        total_num = 0
        correct_num = 0.
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                correct_num = correct_num + float(
                    prediction[batch_index][:, col_index].dot(target[batch_index][:, col_index]))
                total_num = total_num + int(sum(target[batch_index][:, col_index]))
        return correct_num / col_size
    else:
        total_num = 0
        correct_num = 0
        col_size = target.shape[1]
        for col_index in range(col_size):
            correct_num = correct_num + float(prediction[:, col_index].dot(target[:, col_index]))
            total_num = total_num + int(sum(target[:, col_index]))
        return correct_num / col_size


def accuracy_no_other(prediction, target, batch_size=1):  # to be implemented
    if batch_size > 1:
        total_num = 0
        correct_num = 0
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                correct_num = correct_num + float(
                    prediction[batch_index][:-1, col_index].dot(target[batch_index][:-1, col_index]))
                total_num = total_num + int(sum(target[batch_index][:-1, col_index]))
        return correct_num / total_num
    else:
        total_num = 0
        correct_num = 0
        col_size = target.shape[1]
        for col_index in range(col_size):
            correct_num = correct_num + float(prediction[:-1, col_index].dot(target[:-1, col_index]))
            total_num = total_num + int(sum(target[:-1, col_index]))
        return correct_num / total_num


def accuracy_possibility(prediction_poss, target, batch_size=1):
    # to be implemented
    return accuracy(prediction_poss, target, batch_size)


def accuracy_threshold(prediction_poss, target, batch_size=1, threshold=0.05):
    # to be implemented
    accuracy = 0
    if batch_size > 1:
        total_num = 0
        correct_num = 0
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                prob = float(prediction_poss[batch_index][:, col_index].dot(target[batch_index][:, col_index]))
                if prob < threshold:
                    prob = 0
                correct_num = correct_num + prob
                total_num = total_num + int(sum(target[batch_index][:, col_index]))
        accuracy = correct_num / total_num
    else:
        total_num = 0
        correct_num = 0
        col_size = target.shape[1]
        for col_index in range(col_size):
            prob = float(prediction_poss[:, col_index].dot(target[:, col_index]))
            if prob < threshold:
                prob = 0
            correct_num = correct_num + prob
            total_num = total_num + int(sum(target[:, col_index]))
        accuracy = correct_num / total_num
    return accuracy


def pred_catagory_accuracy_maximum(prediction, target, batch_size=1):
    if batch_size > 1:
        accuracy_list = [[0, 0] for x in range(target.shape[1])]
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                # index_ = numpy.flatnonzero(numpy.array(target[batch_index][:, col_index]))
                if int(sum(target[batch_index][:, col_index])) == 1:
                    index = numpy.flatnonzero(numpy.array(prediction[batch_index][:, col_index]))
                    accuracy_list[index[0]][0] += int(target[batch_index][index[0], col_index])
                    accuracy_list[index[0]][1] += 1
        # accuracy = correct_num / total_num
    else:
        accuracy_list = [[0, 0]] * target.shape[0]
        col_size = target.shape[1]
        for col_index in range(col_size):
            # index = numpy.flatnonzero(numpy.array(target[:, col_index]))
            if int(sum(target[:, col_index])) == 1:
                index = numpy.flatnonzero(numpy.array(prediction[:, col_index]))
                accuracy_list[index[0]][0] += int(target[index[0], col_index])
                accuracy_list[index[0]][1] += 1
        # accuracy = correct_num / total_num
    return accuracy_list


def targ_catagory_accuracy_maximum(prediction, target, batch_size=1):
    if batch_size > 1:
        accuracy_list = [[0, 0] for x in range(target.shape[1])]
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                index = numpy.flatnonzero(numpy.array(target[batch_index][:, col_index]))
                if index.size == 1:
                    accuracy_list[index[0]][0] += int(prediction[batch_index][index[0], col_index])
                    accuracy_list[index[0]][1] += 1
        # accuracy = correct_num / total_num
    else:
        accuracy_list = [[0, 0]] * target.shape[0]
        col_size = target.shape[1]
        for col_index in range(col_size):
            index = numpy.flatnonzero(numpy.array(target[:, col_index]))
            if index.size == 1:
                accuracy_list[index[0]][0] += int(prediction[index[0], col_index])
                accuracy_list[index[0]][1] += 1
        # accuracy = correct_num / total_num
    return accuracy_list


def targ_catagory_accuracy_possibility(prediction_poss, target, batch_size=1, threshold=0):
    if batch_size > 1:
        accuracy_list = [[0, 0] for x in range(target.shape[1])]
        col_size = target.shape[2]
        for batch_index in range(batch_size):
            for col_index in range(col_size):
                index = numpy.flatnonzero(numpy.array(target[batch_index][:, col_index]))
                if index.size == 1:
                    prob = float(prediction_poss[batch_index][index[0], col_index])
                    if prob < threshold:
                        prob = 0
                    accuracy_list[index[0]][0] += prob
                    accuracy_list[index[0]][1] += 1
        # accuracy = correct_num / total_num
    else:
        accuracy_list = [[0, 0]] * target.shape[0]
        col_size = target.shape[1]
        for col_index in range(col_size):
            index = numpy.flatnonzero(numpy.array(target[:, col_index]))
            if index.size == 1:
                prob = float(prediction_poss[index[0], col_index])
                if prob < threshold:
                    prob = 0
                accuracy_list[index[0]][0] += prob
                accuracy_list[index[0]][1] += 1
        # accuracy = correct_num / total_num
    return accuracy_list


def compute_accuracy(accuracy_list):
    accuracy = []
    for pair in accuracy_list:
        accuracy.append(float(pair[0]) / pair[1])
    return accuracy


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


def main():

    # dic = defaultdict(lambda: defaultdict(int))
    # dic_no_cut = defaultdict(lambda: defaultdict(int))
    # dic_prediction = defaultdict(lambda: '')
    # train_size = 100000
    # batch_size = 50
    # batch_index = 0
    # while batch_size * batch_index < train_size:
    #     print(batch_index)
    #     input, target = load_data(batch_size=batch_size, batch_index=batch_index)
    #     batch_index += 1
    #     for i in range(len(input)):
    #         #measure_distribution_cut(dic, input[i], target[i])
    #         measure_distribution_no_cut(dic_no_cut, input[i], target[i])
    # # print('cuted columns')
    # # for key in dic.keys():
    # #     # if len(dic[key]) > 1:
    # #     # print('{}:{}'.format(key, dic[key]))
    # #     for label in dic[key].keys():
    # #         if dic[key][label] > 50:
    # #             print(key, label, 'count:{}'.format(dic[key][label]))
    # #         if dic[key][label] / sum(dic[key].values()) > 0.25:
    # #             print(key, label, 'percentage:{}%'.format(dic[key][label] / sum(dic[key].values()) * 100))
    #
    # with open('diction.json', 'w') as fp:
    #     json.dump(dic_no_cut, fp)
    #     print('diction saved')
    #
    # print('table')

    # with open('diction.json', 'r') as fp:
    #     dic_no_cut = json.load(fp)
    #


    pre_acc = 0
    sum = 0

    for key in dic_no_cut.keys():
        if key!='-1,-1,-1,-1,-1,-1,-1,-1,-1,-1':
            max = 0
            for label in dic_no_cut[key].keys():
                sum += dic_no_cut[key][label]
                if dic_no_cut[key][label] > max:
                    max = dic_no_cut[key][label]
                # print(key, label, 'count:{}'.format(dic_no_cut[key][label]))
            pre_acc += max
    print("train accuracy {}".format(pre_acc / sum))

    # with open('diction_prediction.json', 'w') as fp1:
    #     json.dump(dic_prediction, fp1)
    #     print('decision tree saved')
    #
    # # with open('diction_prediction.json', 'r') as f:
    # #     dic_prediction = json.load(f)
    # batch_size = 50
    # batch_index = 2000
    # correct = 0
    # total = 0
    # while batch_size * batch_index < 103000:
    #     print(batch_index)
    #     input, target = load_data(batch_size=batch_size, batch_index=batch_index)
    #     batch_index += 1
    #     for i in range(len(input)):
    #         total += 1
    #         # measure_distribution_cut(dic, input[i], target[i])
    #         input_transformed = input[i].transpose()
    #         target_transformed = target[i].transpose()
    #         key_list = []
    #         value_list = []
    #         for index, row in enumerate(input_transformed):
    #             try:
    #                 i = list(row).index(1)
    #                 t = list(target_transformed[index]).index(1)
    #             except ValueError:
    #                 i = -1
    #                 t = -1
    #             finally:
    #                 key_list.append(str(i))
    #                 value_list.append(str(t))
    #         key = ','.join(key_list)
    #         value = ','.join(value_list)
    #         if dic_prediction[key] == value:
    #             correct += 1
    # print('validation accuracy {}'.format(correct / total))


if __name__ == '__main__':
    # with open('diction.json', 'r') as fp:
    #     dic_no_cut = json.load(fp)
    # with open('diction_prediction.json', 'r') as fp1:
    #     dic_prediction = json.load(fp1)
    # num = 0
    # no_other = 0
    # for key in dic_no_cut.keys():
    #     for label in dic_no_cut[key].keys():
    #         print(label)
    #         label_list = label.split(',')
    #         for l in label_list:
    #             if l!='-1':
    #                 num += dic_no_cut[key][label]
    #                 # if l!='4510':
    #                 #     no_other+= dic_prediction[key][label]
    #
    #
    #         #print(key, label, 'count:{}'.format(dic_no_cut[key][label]))

    # print("train accuracy {}".format(no_other/num))
    print('version 2.2.1')
    sample_print()
    #main()
