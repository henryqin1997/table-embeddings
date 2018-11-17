import json
import os
import numpy
from collections import defaultdict
#from etl import Table
#from .train import one_hot

training_data_dir = 'data/train'
training_files_json = 'data/training_files_filtered.json'
#training_files = json.load(open(training_files_json))
testing_data_dir = 'data/sample_random_label_test'
activate_data_dir = 'data/sample_random_label'
# testing_files_json = 'data/testing_files_random_label.json'
# testing_files = [[y for y in json.load(open(testing_files_json)) if y[0] == str(x)] for x in range(10)]
tag_to_index = {'LOCATION': 0, 'PERSON': 1, 'ORGANIZATION': 2, 'MONEY': 3, 'PERCENT': 4, 'DATE': 5, 'TIME': 6}


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

        inputs_transformed.append(numpy.array([one_hot(row) for row in input.transpose()]).transpose())
        targets_transformed.append(target)
    return numpy.array(inputs_transformed), numpy.array(targets_transformed)

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

def main():
    print('decision tree v2.2')
    dic_no_cut = defaultdict(lambda: defaultdict(int))
    #dic_no_cut = json.load(open('diction_with0.json'))
    dic = defaultdict(lambda: defaultdict(int))
    dic_prediction = defaultdict(lambda: '')
    train_size = 100000
    batch_size = 50
    batch_index = 0
    while batch_size * batch_index < train_size:
        print(batch_index)
        input, target = load_data(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1
        for i in range(len(input)):
            measure_distribution_cut(dic, input[i], target[i])
            measure_distribution_no_cut(dic_no_cut, input[i], target[i])


        # for key in dic.keys():
        #     # if len(dic[key]) > 1:
        #     # print('{}:{}'.format(key, dic[key]))
        #     for label in dic[key].keys():
        #         if dic[key][label] > 50:
        #             print(key, label, 'count:{}'.format(dic[key][label]))
        #         if dic[key][label] / sum(dic[key].values()) > 0.25:
        #             print(key, label, 'percentage:{}%'.format(dic[key][label] / sum(dic[key].values()) * 100))



    with open('dic_cut_with0.json','w') as fp:
        json.dump(dic,fp)
        print('dic_cut_with0 saved')

    with open('diction_with0.json', 'w') as fp:
        json.dump(dic_no_cut, fp)
        print('diction saved')

    print('table')

    pre_acc = 0
    sum = 0

    for key in dic_no_cut.keys():
        max = 0
        maxlabel = ''
        for label in dic_no_cut[key].keys():
            if label!='-1,-1,-1,-1,-1,-1,-1,-1,-1,-1':
                sum += dic_no_cut[key][label]
                if dic_no_cut[key][label] > max:
                    max = dic_no_cut[key][label]
                    maxlabel = label
                # print(key, label, 'count:{}'.format(dic_no_cut[key][label]))
        pre_acc += max
        dic_prediction[key] = maxlabel
    print("train accuracy {}".format(pre_acc / sum))

    with open('diction_prediction_with0.json', 'w') as fp:
        json.dump(dic_prediction, fp)
        print('diciton prediction saved')

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
            # measure_distribution_cut(dic, input[i], target[i])
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


if __name__ == '__main__':
    main()
    with open('dic_cut_with0.json','r') as fp:
        dic_cut=json.load(fp)
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
    with open('dic_cut_pred.json','w') as fp:
        json.dump(dic_cut_pred,fp)