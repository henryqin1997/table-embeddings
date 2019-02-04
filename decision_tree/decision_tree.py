import json
from .load_ner import load_sample_random_label,load_sample_random_table
from collections import defaultdict

training_data_dir = 'data/train'
training_files_json = 'data/training_files.json'
training_files = json.load(open(training_files_json))
testing_data_random_label_dir = 'data/sample_random_label_test'
activate_data_random_label_dir = 'data/sample_random_label'
testing_files_random_label_json = 'data/testing_files_random_label.json'
testing_files_random_label = [[y for y in json.load(open(testing_files_random_label_json)) if y[0] == str(x)] for x in range(10)]
testing_data_random_table_dir = 'data/sample_random_table_test'
activate_data_random_table_dir = 'data/sample_random_table'
testing_files_random_table_json = 'data/testing_files_random_table.json'
testing_files_random_table = [[y for y in json.load(open(testing_files_random_table_json)) if y[0] == str(x)] for x in range(1)]
tag_to_index = {'LOCATION': 0, 'PERSON': 1, 'ORGANIZATION': 2, 'MONEY': 3, 'PERCENT': 4, 'DATE': 5, 'TIME': 6}

def qin_similarity(list1, list2):
    intersection=0
    union=max(len(list1),len(list2))
    for i in range(min(len(list1),len(list2))):
        if list1[i]==list2[i] and list1[i]!=-1:
            intersection+=1
    return float(intersection / union)

def li_similarity(list1,list2):
    count=0
    for i in range(len(list1)):
        if list1[i]==list2[i]:
            count+=1
    return float(count/len(list1))

def diction_pred(dic,dic_cut,feature):
    maxkey=''
    maxsim=0
    feature_processed=[int(x) for x in feature if int(x)!=-1]
    for key in dic.keys():
        key_processed = [int(x) for x in key.split(',') if x!='-1']
        sim = qin_similarity(feature_processed,key_processed)
        if sim>maxsim:
            maxkey=key
            maxsim=sim

    pred = dic[maxkey]
    pred = [int(x) for x in pred.split(',')]
    maxkey = [int(x) for x in maxkey.split(',')]

    for i in range(len(maxkey)):
        if feature[i]!=-1:
            if maxkey[i]!=feature[i]:
                if dic_cut[str(feature[i])][1]>0.5:
                    pred[i]=int(dic_cut[str(feature[i])][0])
        else:
            break

    return pred

def diction_pred_advanced(dic,feature):
    maxkey = ''
    maxsim = 0
    feature_processed = [int(x) for x in feature if int(x) != -1]
    for key in dic.keys():
        key_processed = [int(x) for x in key.split(',')]
        sim = li_similarity(feature_processed, key_processed)
        if sim > maxsim:
            maxkey = key
            maxsim = sim

    pred = dic[maxkey][0]
    pred = [int(x) for x in pred.split(',')]

    return pred

def sample_dict(sample_data,sample_summary,missed_feature,faultdic,prediction):
    batch_size=len(sample_data)
    #missed_feature=[]
    #sample_summary = defaultdict(lambda: [0, 0])
    # for batch in range(iteration):
    #     sample_summary=defaultdict(lambda:[0,0])
    #     features=sample_feature[batch*batch_size:batch_size*(batch+1)]
    #     targets=sample_target[batch*batch_size:batch_size*(batch+1)]
    #     actives=sample_active[batch*batch_size:batch_size*(batch+1)]

    dic_cut_pred = json.load(open('decision_tree/dic_cut_pred.json'))
    for index in range(batch_size):
        feature=sample_data[index][0]
        target=sample_data[index][1]
        activate=sample_data[index][2]
        if ','.join(str(x) for x in feature) not in prediction:
            pred = diction_pred(prediction,dic_cut_pred,feature)
            for i in range(10):
                if activate[i]==1 and target[i]!=-1:
                    sample_summary[target[i]][1]+=1
                    if int(pred.split(',')[i])==target[i]:
                        sample_summary[target[i]][0] += 1

            missed_feature.add(','.join(str(x) for x in feature))
        else:
            mis=False
            for i in range(10):
                if activate[i]==1 and target[i]!=-1:
                    sample_summary[target[i]][3]+=1
                    if target[i]==int(prediction[','.join(str(x) for x in feature)].split(',')[i]):
                        sample_summary[target[i]][2]+=1
                    else:
                        mis=True
            if mis:
                mispred = [str(x) for x in target]
                for i in range(10):
                    if activate[i]==1 and target[i]!=prediction[','.join(str(x) for x in feature)].split(',')[i]:
                        mispred[i]+='*'
                mispred=','.join(mispred)
                faultdic[','.join(str(x) for x in feature)].append([prediction[','.join(str(x) for x in feature)],mispred])

def sample_dict_table(sample_data,sample_summary,missed_feature,faultdic,prediction):
    batch_size=len(sample_data)
    dic_cut = json.load(open('decision_tree/dic_cut_pred.json'))
    for index in range(batch_size):
        feature=sample_data[index][0]
        target=sample_data[index][1]
        if ','.join(str(x) for x in feature) not in prediction:
            pred = diction_pred(prediction,dic_cut,feature)
            for i in range(10):
                if target[i]!=-1:
                    sample_summary[target[i]][1]+=1
                    if int(pred.split(',')[i])==target[i]:
                        sample_summary[target[i]][0] += 1
            missed_feature.add(','.join(str(x) for x in feature))
        else:
            mis=False
            for i in range(10):
                if target[i]!=-1:
                    sample_summary[target[i]][3]+=1
                    if target[i]==int(prediction[','.join(str(x) for x in feature)].split(',')[i]):
                        sample_summary[target[i]][2]+=1
                    else:
                        mis=True
            if mis:
                mispred = [str(x) for x in target]
                for i in range(10):
                    if target[i]!=-1 and target[i]!=prediction[','.join(str(x) for x in feature)].split(',')[i]:
                        mispred[i]+='*'
                mispred=','.join(mispred)
                faultdic[','.join(str(x) for x in feature)].append([prediction[','.join(str(x) for x in feature)],mispred])

def sample_dict_advanced(sample_data,sample_summary,missed_feature,faultdic,prediction):
    batch_size=len(sample_data)
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
            mis=False
            for i in range(10):
                if activate[i]==1 and target[i]!=-1:
                    sample_summary[target[i]][3]+=1
                    if target[i]==int(prediction[','.join(str(x) for x in feature)][0].split(',')[i]):
                        sample_summary[target[i]][2]+=1
                    else:
                        mis=True
            if mis:
                mispred = [str(x) for x in target]
                for i in range(10):
                    if activate[i]==1 and target[i]!=prediction[','.join(str(x) for x in feature)][0].split(',')[i]:
                        mispred[i]+='*'
                mispred=','.join(mispred)
                faultdic[','.join(str(x) for x in feature)].append([prediction[','.join(str(x) for x in feature)][0],mispred])

def sample_dict_table_advanced(sample_data,sample_summary,missed_feature,faultdic,prediction):
    batch_size=len(sample_data)
    for index in range(batch_size):
        feature=sample_data[index][0]
        target=sample_data[index][1]
        if ','.join(str(x) for x in feature) not in prediction:
            pred = diction_pred(prediction,feature)
            for i in range(10):
                if target[i]!=-1:
                    sample_summary[target[i]][1]+=1
                    if int(pred.split(',')[i])==target[i]:
                        sample_summary[target[i]][0] += 1
            missed_feature.add(','.join(str(x) for x in feature))
        else:
            mis=False
            for i in range(10):
                if target[i]!=-1:
                    sample_summary[target[i]][3]+=1
                    if target[i]==int(prediction[','.join(str(x) for x in feature)][0].split(',')[i]):
                        sample_summary[target[i]][2]+=1
                    else:
                        mis=True
            if mis:
                mispred = [str(x) for x in target]
                for i in range(10):
                    if target[i]!=-1 and target[i]!=prediction[','.join(str(x) for x in feature)][0].split(',')[i]:
                        mispred[i]+='*'
                mispred=','.join(mispred)
                faultdic[','.join(str(x) for x in feature)].append([prediction[','.join(str(x) for x in feature)][0],mispred])

def sample_print():
    batch_size=50
    sample_size=4500
    missed_feature=set([])
    faultdic = defaultdict(lambda: [])
    with open('decision_tree/diction_prediction_with0.json', 'r') as fp:
        prediction = json.load(fp)
    print(len(prediction))
    for sample_index in range(10):
        sample_summary = defaultdict(lambda: [0, 0, 0, 0])
        batch_index = 0
        while batch_size*batch_index<sample_size:
            sample_data=load_sample_random_label(sample_index,batch_size,batch_index)
            sample_dict(sample_data,sample_summary,missed_feature,faultdic,prediction)
            batch_index+=1
        with open("decision_tree/sample_dict_it={}".format(sample_index), 'w') as wfp:
            json.dump(sample_summary, wfp)

    sample_table_summary = defaultdict(lambda: [0, 0, 0, 0])
    faultdic2=defaultdict(lambda: [])
    batch_index = 0
    while batch_size * batch_index < 9000:
        sample_data = load_sample_random_table(0, batch_size, batch_index)
        sample_dict_table(sample_data, sample_table_summary, missed_feature, faultdic2, prediction)
        batch_index += 1

    with open("decision_tree/sample_table_pred", 'w') as wfp:
        json.dump(sample_table_summary, wfp)
    with open('decision_tree/fault_table.json','w') as wfp:
        json.dump(faultdic2,wfp)

    with open("decision_tree/miss_features",'w') as wfp:
        for f in missed_feature:
            wfp.write(f+"\n")
    with open("decision_tree/fault_diction_table.json",'w') as wfp:
        json.dump(faultdic,wfp)

def sample_print_advanced():
    batch_size = 50
    sample_size = 4500
    missed_feature = set([])
    faultdic = defaultdict(lambda: [])
    with open('decision_tree/diction_prediction_advanced.json', 'r') as fp:
        prediction = json.load(fp)
    print(len(prediction))
    for sample_index in range(10):
        sample_summary = defaultdict(lambda: [0, 0, 0, 0])
        batch_index = 0
        while batch_size * batch_index < sample_size:
            sample_data = load_sample_random_label(sample_index, batch_size, batch_index)
            sample_dict_advanced(sample_data, sample_summary, missed_feature, faultdic, prediction)
            batch_index += 1
        with open("decision_tree/sample_dict_it={}".format(sample_index), 'w') as wfp:
            json.dump(sample_summary, wfp)

    sample_table_summary = defaultdict(lambda: [0, 0, 0, 0])
    faultdic2 = defaultdict(lambda: [])
    batch_index = 0
    while batch_size * batch_index < 9000:
        sample_data = load_sample_random_table(0, batch_size, batch_index)
        sample_dict_table_advanced(sample_data, sample_table_summary, missed_feature, faultdic2, prediction)
        batch_index += 1

    with open("decision_tree/sample_table_pred", 'w') as wfp:
        json.dump(sample_table_summary, wfp)
    with open('decision_tree/fault_table.json', 'w') as wfp:
        json.dump(faultdic2, wfp)

    with open("decision_tree/miss_features", 'w') as wfp:
        for f in missed_feature:
            wfp.write(f + "\n")
    with open("decision_tree/fault_diction_table.json", 'w') as wfp:
        json.dump(faultdic, wfp)

def label_num_str(labels):
    labels = [int(x) for x in labels.split(',') if x!=-1]
    return len(labels)
def label_num_str_no_other(labels):
    labels = [int(x) for x in labels.split(',') if x != -1 and x!=3333]
    return len(labels)

def label_num_arr(labels):
    return len([x for x in labels if x!=-1])

def correct_pred(pred, labels):
    pred = [int(x) for x in pred.split(',')]
    labels = [int(x) for x in labels.split(',')]
    count = 0
    for i in range(min(len(pred),len(labels))):
        if labels[i]!=-1:
            if pred[i]==labels[i]:
                count+=1
    return count

def correct_pred_no_other(pred, labels):
    pred = [int(x) for x in pred.split(',')]
    labels = [int(x) for x in labels.split(',')]
    count = 0
    for i in range(min(len(pred),len(labels))):
        if labels[i]!=-1 and labels[i]!=3333:
            if pred[i]==labels[i]:
                count+=1
    return count
