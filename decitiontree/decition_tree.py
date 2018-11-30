import json
from load import load_sample_random_label,load_sample_random_table,load_data
from collections import defaultdict

training_data_dir = 'data/train'
training_files_json = 'data/training_files_filtered.json'
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

def diction_pred(dic,feature):
    maxkey=''
    maxsim=0
    feature_processed=[x for x in feature if x!=-1]
    for key in dic.keys():
        key_processed = [int(x) for x in key.split(',') if x!='-1']
        sim = qin_similarity(feature_processed,key_processed)
        if sim>maxsim:
            maxkey=key
            maxsim=sim
    return [int(x) for x in maxkey.split(',')],dic[maxkey]

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
            sel_feature, pred = diction_pred(prediction,feature)
            for i in range(10):
                if activate[i]==1 and target[i]!=-1:
                    sample_summary[target[i]][1]+=1
                    if feature[i]==sel_feature[i]:
                        if int(pred.split(',')[i])==target[i]:
                            sample_summary[target[i]][0] += 1
                    else:
                        dic_cut=json.load(open('decitiontree/dic_cut_pred.json'))
                        if str(feature[i]) in dic_cut.keys():
                            if dic_cut[str(feature[i])][1]>=0.5:
                                if int(dic_cut[str(feature[i])][0])==target[i]:
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
                    if activate[i]==1:
                        mispred[i]+='*'
                mispred=','.join(mispred)
                faultdic[','.join(str(x) for x in feature)].append([prediction[','.join(str(x) for x in feature)],mispred])

def sample_dict_table(sample_data,sample_summary,missed_feature,faultdic,prediction):
    batch_size=len(sample_data)
    dic_cut = json.load(open('decitiontree/dic_cut_pred.json'))
    for index in range(batch_size):
        feature=sample_data[index][0]
        target=sample_data[index][1]
        if ','.join(str(x) for x in feature) not in prediction:
            sel_feature, pred = diction_pred(prediction,feature)
            for i in range(10):
                if target[i]!=-1:
                    sample_summary[target[i]][1]+=1
                    if feature[i]==sel_feature[i]:
                        if int(pred.split(',')[i])==target[i]:
                            sample_summary[target[i]][0] += 1
                    else:
                        if str(feature[i]) in dic_cut.keys():
                            if dic_cut[str(feature[i])][1]>=0.5:
                                if int(dic_cut[str(feature[i])][0])==target[i]:
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


def sample_print():
    batch_size=50
    sample_size=4500
    missed_feature=set([])
    faultdic = defaultdict(lambda: [])
    with open('decitiontree/diction_prediction.json', 'r') as fp:
        prediction = json.load(fp)
    print(len(prediction))
    for sample_index in range(10):
        sample_summary = defaultdict(lambda: [0, 0, 0, 0])
        batch_index = 0
        while batch_size*batch_index<sample_size:
            sample_data=load_sample_random_label(sample_index,batch_size,batch_index)
            sample_dict(sample_data,sample_summary,missed_feature,faultdic,prediction)
            batch_index+=1
        with open("decitiontree/sample_dict_it={}".format(sample_index), 'w') as wfp:
            json.dump(sample_summary, wfp)

    sample_table_summary = defaultdict(lambda: [0, 0, 0, 0])
    faultdic2=defaultdict(lambda: [])
    batch_index = 0
    while batch_size * batch_index < 9000:
        sample_data = load_sample_random_table(0, batch_size, batch_index)
        sample_dict_table(sample_data, sample_table_summary, missed_feature, faultdic2, prediction)
        batch_index += 1

    with open("decitiontree/sample_table_pred", 'w') as wfp:
        json.dump(sample_table_summary, wfp)
    with open('decitiontree/fault_table.json','w') as wfp:
        json.dump(faultdic2,wfp)

    with open("decitiontree/miss_features",'w') as wfp:
        for f in missed_feature:
            wfp.write(f+"\n")
    with open("decitiontree/fault_diction_table.json",'w') as wfp:
        json.dump(faultdic,wfp)