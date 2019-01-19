import numpy as np
from numerical_space.load import load
import itertools
import math
import os
import json
from collections import defaultdict
from operator import itemgetter

filter_lambdas = [lambda l: not l[6] and l[5] == 1, lambda l: l[6] and l[5] == 1, lambda l: not l[6] and l[5] == 0,
                  lambda l: l[6] and l[5] == 0, lambda l: not l[6] and l[5] == -1, lambda l: l[6] and l[5] == -1]
filter_names = ['asc int', 'asc float', 'rand int', 'rand float', 'desc int', 'desc float']
wordlist = json.load(open('data/wordlist_v6_index.json'))


def falconn_test():
    train_size = 103000
    batch_size = 50

    print('Load 103k data')
    if os.path.isfile('numerical_space/dataset.csv'):
        dataset = np.genfromtxt('numerical_space/dataset.csv', delimiter=',')
    else:
        summary = []
        for batch_index in range(int(math.floor(train_size / batch_size))):
            print(batch_index)
            summary.append(load(batch_size=batch_size, batch_index=batch_index))

        dataset = np.array(list(itertools.chain.from_iterable(itertools.chain.from_iterable(summary))))
        np.savetxt('numerical_space/dataset.csv', dataset, delimiter=',')

    # Filter out mean, variance, min, max all zero
    dataset = list(filter(lambda l: np.linalg.norm(l[1:5]) > 0, dataset))

    for filter_lambda, filter_name in zip(filter_lambdas, filter_names):
        print('Predict ' + filter_name)
        dataset_filtered = list(filter(filter_lambda, dataset))
        vectors = np.array([l[1:5] for l in dataset_filtered])

        number_of_queries = int(round(len(vectors) * 0.1))
        print('Vector length: {}, Number of queries: {}'.format(len(vectors), number_of_queries))

        if number_of_queries == 0:
            continue

        vectors /= np.linalg.norm(vectors, axis=1).reshape(-1, 1)
        labels = [l[0] for l in dataset_filtered]

        answers = []
        for k, query in enumerate(vectors[-number_of_queries:]):
            answers.append(labels[np.dot(vectors[:-number_of_queries], query).argmax()])
            # if k % 100 == 9:
            #     print('{}/{} - Accuracy: {}'.format(
            #         k + 1,
            #         number_of_queries,
            #         vector_accuracy(answers,
            #                         labels[len(labels) - number_of_queries:len(labels) - number_of_queries + k + 1])))

        print('Accuracy {}: {}'.format(filter_name, vector_accuracy(labels[-number_of_queries:], answers)))
        print('No other accuracy {}: {}'.format(filter_name,
                                                vector_accuracy_no_other(labels[-number_of_queries:], answers)))
        generate_report(labels[-number_of_queries:], answers,
                        'numerical_space/predict_{}.json'.format(filter_name.replace(' ', '_')))


def vector_accuracy(a, b):
    assert len(a) == len(b)
    return sum(i == j for i, j in zip(a, b)) / len(a)


def vector_accuracy_no_other(a, b, other=3333):
    no_other = list(filter(lambda item: item[0] != other, zip(a, b)))
    return sum(i == j for i, j in no_other) / len(no_other)


def get_label_by_index(index):
    try:
        return list(wordlist.items())[int(index)][0]
    except IndexError:
        return 'OTHER'


def generate_report(a, b, filename, other=3333):
    report = {}
    for i, j in zip(a, b):
        if i != other:
            label = get_label_by_index(i)
            if label not in report:
                report[label] = defaultdict(int)
            report[label][get_label_by_index(j)] += 1
    for label in report:
        report[label] = dict(sorted(report[label].items(), key=itemgetter(1), reverse=True))
    report = dict(sorted(report.items(), key=lambda item: sum(item[1].values()), reverse=True))
    json.dump(report, open(filename, 'w+'), indent=4)


if __name__ == '__main__':
    falconn_test()
