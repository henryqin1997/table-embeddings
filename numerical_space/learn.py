'''
To build a space of numerical data points to predict label of them.
'''
from numerical_space.load import load
from collections import defaultdict
import json

# For debug
training_data_dir = 'data/train_100_sample'
training_files_json = 'data/training_files_100_sample.json'

def build_space():
    '''
    Build 6 diction, match (features)->(label,info), named numerical_dic_1_float
    features--mean, variance, min, max, is_ordered (1 ascending, 0 random, -1 descending), is_float (True/False)
    '''
    dic_asc_fl = defaultdict(lambda:defaultdict(int))
    dic_ran_fl = defaultdict(lambda:defaultdict(int))
    dic_des_fl = defaultdict(lambda:defaultdict(int))
    dic_asc_int = defaultdict(lambda:defaultdict(int))
    dic_ran_int = defaultdict(lambda:defaultdict(int))
    dic_des_int = defaultdict(lambda:defaultdict(int))
    train_size = 100000
    batch_size = 50
    batch_index = 0
    while batch_size * batch_index < train_size:
        print(batch_index)
        summary = load(batch_size=batch_size, batch_index=batch_index)
        for table_summary in summary:
            for column_summary in table_summary:
                feature_key=','.join(map(str,column_summary[1:5]))
                label_info_key=column_summary[0]
                if column_summary[6]:
                    if column_summary[5]==1:
                        dic_asc_fl[feature_key][label_info_key]+=1
                    elif column_summary[5]==0:
                        dic_ran_fl[feature_key][label_info_key]+=1
                    else:
                        dic_des_fl[feature_key][label_info_key]+=1
                else:
                    if column_summary[5]==1:
                        dic_asc_int[feature_key][label_info_key]+=1
                    elif column_summary[5]==0:
                        dic_ran_int[feature_key][label_info_key]+=1
                    else:
                        dic_des_int[feature_key][label_info_key]+=1
        batch_index += 1

    with open('numerical_space/numerical_dic_1_float.json','w') as wfp:
        json.dump(dic_asc_fl,wfp)
    with open('numerical_space/numerical_dic_0_float.json','w') as wfp:
        json.dump(dic_ran_fl,wfp)
    with open('numerical_space/numerical_dic_-1_float.json','w') as wfp:
        json.dump(dic_des_fl,wfp)
    with open('numerical_space/numerical_dic_1_int.json','w') as wfp:
        json.dump(dic_asc_int,wfp)
    with open('numerical_space/numerical_dic_0_int.json','w') as wfp:
        json.dump(dic_ran_int,wfp)
    with open('numerical_space/numerical_dic_-1_int.json','w') as wfp:
        json.dump(dic_des_int,wfp)

if __name__ == '__main__':
    build_space()


