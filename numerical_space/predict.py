from .load import load
import json

def predict(summary):


def main():
    dic_asc_fl = json.load(open('numerical_dic_1_float.json'))
    dic_ran_fl = json.load(open('numerical_dic_0_float.json'))
    dic_des_fl = json.load(open('numerical_dic_-1_float.json'))
    dic_asc_int = json.load(open('numerical_dic_1_float.json'))
    dic_ran_int = json.load(open('numerical_dic_0_float.json'))
    dic_des_int = json.load(open('numerical_dic_-1_float.json'))
    train_size = 100000
    batch_size = 50
    batch_index = 0
    while batch_size * batch_index < train_size:
        print(batch_index)
        summary = load(batch_size=batch_size, batch_index=batch_index)
        for table_summary in summary:
            for column_summary in table_summary: