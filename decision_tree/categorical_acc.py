import json
from collections import defaultdict
from .decision_tree import diction_pred
from .decision_tree import label_num_arr, label_num_str, label_num_str_no_other, correct_pred, correct_pred_no_other

def cal_accuracy():
    dic_no_cut = json.load(open('decision_tree/diction_nst{}.json'.format("10digits")))
    dic_prediction = json.load(open('decision_tree/diction_nst_prediction{}.json'.format("10digits")))
    dic_cut_pred = json.load(open('decision_tree/dic_nst_cut_pred{}.json'.format("10digits")))

    # dic_no_cut = json.load(open('decision_tree/diction.json'))
    # dic_prediction = json.load(open('decision_tree/diction_prediction_with0.json'))
    # dic_cut_pred = json.load(open('decision_tree/dic_cut_pred.json'))

    categorical_acc = defaultdict(lambda: [0,0,0])

    # pre_acc = 0
    # sum = 0
    # sum_no_other = 0
    # acc_no_other = 0

    for key in dic_no_cut.keys():
        max = 0
        maxlabel = ''

        #pred = ','.join([str(x) for x in diction_pred(dic_prediction, dic_cut_pred, key.split(','))])
        for label in dic_no_cut[key].keys():
            if label != '-1,-1,-1,-1,-1,-1,-1,-1,-1,-1':

                # sum += dic_no_cut[key][label] * label_num_str(label)
                # sum_no_other += dic_no_cut[key][label] * label_num_str_no_other(label)

                cats = label.split(',')
                for cat in cats:
                    if cat!='-1':
                        categorical_acc[cat][0]+=dic_no_cut[key][label]

                if dic_no_cut[key][label] > max:
                    max = dic_no_cut[key][label]
                    maxlabel = label
                print(key, label, 'count:{}'.format(dic_no_cut[key][label]))
            else:
                continue

        cat_max = maxlabel.split(',')

        for label in dic_no_cut[key].keys():
            # pre_acc += dic_no_cut[key][label] * correct_pred(maxlabel, label)
            # acc_no_other += dic_no_cut[key][label] * correct_pred_no_other(maxlabel, label)
            cats=label.split(',')
            for i,cat in enumerate(cats):
                if cat != '-1':
                    if cat_max[i]==cat:
                        categorical_acc[cat][1]+=dic_no_cut[key][label]

    for key in categorical_acc.keys():
        categorical_acc[key][2]=categorical_acc[key][1]/categorical_acc[key][0]


    # print("train accuracy {}".format(pre_acc / sum))
    # print("train accuracy no other {}".format(acc_no_other / sum_no_other))

    with open('decision_tree/categorical_acc_previous.json', 'w') as fp:
        json.dump(categorical_acc, fp, indent=4)


def cal_head_acc():
    #categorical_acc = json.load(open('decision_tree/categorical_acc_previous.json','r'))
    categorical_acc = json.load(open('decision_tree/categorical_acc.json', 'r'))
    total_num = 0
    total_correct = 0
    for i in range(3333):
        if str(i) in categorical_acc:
            total_num+=categorical_acc[str(i)][0]
            total_correct+=categorical_acc[str(i)][1]
    print('head_acc is: {}'.format(total_correct/total_num))

if __name__=='__main__':
    cal_accuracy()
    #cal_head_acc()