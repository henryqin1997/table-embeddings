from decision_tree.load import load_data_with_raw
import json
import os
from etl.table import Table

keycolumn = 'date'


def find_all():
    wordlist = json.load(open('data/wordlist_v6_index.json'))
    key_index = wordlist[keycolumn]

    if not os.path.exists('tools/{}'.format(keycolumn)):
        os.makedirs('tools/{}'.format(keycolumn))

    wfp = open('tools/{}/columns.txt'.format(keycolumn), 'w')

    size = 100000
    batch_size = 50
    batch_index = 0

    while batch_size * batch_index < size:
        print(batch_index)
        raw, input, target = load_data_with_raw(batch_size=batch_size, batch_index=batch_index)
        batch_index += 1

        for i in range(len(target)):
            for j in range(len(target[i])):
                if target[i][j] == key_index:
                    table = Table(raw[i])
                    wfp.write('{}\n'.format(table.get_attributes()[j]))
                elif target[i][j] == -1:
                    break
    wfp.close()


if __name__ == '__main__':
    find_all()
