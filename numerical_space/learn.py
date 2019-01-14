'''
To build a space of numerical data points to predict label of them.
'''
from .load import load

def build_space():
    '''Build 6 diction, match (features)->(label,info), named numericall_dic_1_float'''

    train_size = 100000
    batch_size = 50
    batch_index = 0
    while batch_size * batch_index < train_size:
        print(batch_index)
        summary = load(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1




