import json
import os
import operator
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from .table import Table, satisfy_variants

input_dir = 'data/input'

if __name__ == '__main__':
    x = []
    y = []
    table_num = 0
    word_num = 0
    word_count = {}
    files = os.listdir(input_dir)
    for file in files:
        with open(os.path.join(input_dir, file), encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data = json.loads(line)
                if satisfy_variants(data):
                    table_num += 1
                    table = Table(data)
                    words = table.get_header()
                    for word in words:
                        word = word.lower().strip()
                        if len(word) > 2:
                            if word in word_count:
                                word_count[word] += 1
                            else:
                                word_num += 1
                                x.append(table_num)
                                y.append(word_num)
                                word_count[word] = 1
    word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    print(word_count)
    plt.plot(x, y)
    plt.savefig('etl/wordlist.png')
