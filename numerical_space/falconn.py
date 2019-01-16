import numpy as np
import falconn
import timeit
from numerical_space.load import load
import itertools
import json


def falconn_test():
    train_size = 100
    batch_size = 50
    batch_index = 0
    while batch_size * batch_index < train_size:
        print(batch_index)
        summary = load(batch_size=batch_size, batch_index=batch_index)
        print('---')
        print(list(itertools.chain.from_iterable(summary)))
        print('---')
        for table_summary in summary:
            print(table_summary)
            for column_summary in table_summary:
                print(column_summary)
                # feature_key=','.join(map(str,column_summary[1:5]))
                # label_info_key=column_summary[0]
                # if column_summary[6]:
                #     if column_summary[5]==1:
                #         dic_asc_fl[feature_key][label_info_key]+=1
                #     elif column_summary[5]==0:
                #         dic_ran_fl[feature_key][label_info_key]+=1
                #     else:
                #         dic_des_fl[feature_key][label_info_key]+=1
                # else:
                #     if column_summary[5]==1:
                #         dic_asc_int[feature_key][label_info_key]+=1
                #     elif column_summary[5]==0:
                #         dic_ran_int[feature_key][label_info_key]+=1
                #     else:
                #         dic_des_int[feature_key][label_info_key]+=1
        batch_index += 1

if __name__ == '__main__':
    falconn_test()


