import numpy as np
from numerical_space.load import load
import itertools
import math
import os

filter_lambdas = [lambda l: not l[6] and l[5] == 1, lambda l: l[6] and l[5] == 1, lambda l: not l[6] and l[5] == 0,
                  lambda l: l[6] and l[5] == 0, lambda l: not l[6] and l[5] == -1, lambda l: l[6] and l[5] == -1]
filter_names = ['asc int', 'asc float', 'rand int', 'rand float', 'desc int', 'desc float']


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


def vector_accuracy(a, b):
    assert len(a) == len(b)
    return sum(i == j for i, j in zip(a, b)) / len(a)


def vector_accuracy_no_other(a, b, other=3333):
    no_other = list(filter(lambda item: item[0] != other, zip(a, b)))
    return sum(i == j for i, j in no_other) / len(no_other)


if __name__ == '__main__':
    falconn_test()
