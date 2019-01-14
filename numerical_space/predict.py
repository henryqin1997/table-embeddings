from .load import load
import json

def predict(summary):
    '''Do some search, find closet point for the given point.  Problem: Cosine or L2.'''
    fori = 'float' if summary[-1]==1 else 'int'
    dicname = 'numerical_dic_{}_{}.json'.format(summary[-1],fori)
    dic = json.load(open(dicname))


    return "0"



def main():
    dic_asc_fl = json.load(open('numerical_dic_1_float.json'))
    dic_ran_fl = json.load(open('numerical_dic_0_float.json'))
    dic_des_fl = json.load(open('numerical_dic_-1_float.json'))
    dic_asc_int = json.load(open('numerical_dic_1_int.json'))
    dic_ran_int = json.load(open('numerical_dic_0_int.json'))
    dic_des_int = json.load(open('numerical_dic_-1_int.json'))
    train_size = 103000
    batch_size = 50
    batch_index = 2000
    while batch_size * batch_index < train_size:
        print(batch_index)
        summary = load(batch_size=batch_size, batch_index=batch_index)
        for table_summary in summary:
            for column_summary in table_summary:
                pred = predict(column_summary)




if __name__=='main':
    main()