import json
import os
import operator
import random
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .table import Table, satisfy_variants

input_dir = 'data/input_old'

if __name__ == '__main__':
    x = []
    y = []
    x2 = []
    y2 = [[], [], [], []]
    x3 = []
    y3 = [[], [], []]
    table_num = 0
    word_num = 0
    word_count = {}
    lines = []
    files = os.listdir(input_dir)
    for file in files:
        with open(os.path.join(input_dir, file), encoding='utf-8') as f:
            lines += f.readlines()
    random.Random(10).shuffle(lines)

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
                        word_count[word] = 1
                        word_num += 1
                        x.append(table_num)
                        y.append(word_num)

            if table_num % 10000 == 0:
                x2.append(table_num)
                y2[0].append(len(list(filter(lambda item: item[1] == 1, word_count.items()))))
                y2[1].append(len(
                    list(filter(lambda item: 1 < item[1] < 10, word_count.items()))))
                y2[2].append(len(
                    list(filter(lambda item: 10 <= item[1] < 50, word_count.items()))))
                y2[3].append(len(
                    list(filter(lambda item: 50 <= item[1], word_count.items()))))

                x3.append(table_num)
                new0 = sum(
                    list(map(lambda item: item[1], filter(lambda item: 50 <= item[1], word_count.items()))))
                y3[0].append(new0)
                new1 = sum(word_count.values())
                y3[1].append(new1)
                y3[2].append(new0 / new1)

    word_count = dict(sorted(word_count.items(), key=operator.itemgetter(1), reverse=True))
    json.dump(word_count, open('data/wordlist_v3.json', 'w'), indent=4)
    plt.plot(x, y)
    plt.xlabel('# Tables')
    plt.ylabel('# Labels')
    plt.savefig('etl/wordlist.png')

    plt.clf()
    print(x2)
    print(y2)
    width = 7500
    p1 = plt.bar(x2, y2[0], width, color='r')
    p2 = plt.bar(x2, y2[1], width, bottom=y2[0], color='b')
    p3 = plt.bar(x2, y2[2], width,
                 bottom=np.array(y2[0]) + np.array(y2[1]), color='g')
    p4 = plt.bar(x2, y2[3], width,
                 bottom=np.array(y2[0]) + np.array(y2[1]) + np.array(y2[2]),
                 color='c')
    plt.xlabel('# Tables')
    plt.ylabel('# Labels')
    plt.legend((p1[0], p2[0], p3[0], p4[0]),
               ('1', '(1,10)', '[10,50)', '[50,)'), fontsize=8, ncol=4, framealpha=0, fancybox=True)
    plt.savefig('etl/wordlist_bar.png')

    plt.clf()
    print(x3)
    print(y3)
    p1 = plt.plot(x3, y3[2], color='r')
    plt.xlabel('# Tables')
    plt.ylabel('Top 1k Fraction')
    plt.ylim([0, 1])
    plt.savefig('etl/wordlist_1k_fraction.png')
