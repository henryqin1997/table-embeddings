import numpy as np
from numerical_space.load import load
import itertools
import math


def falconn_test():
    train_size = 103000
    batch_size = 50
    number_of_queries = 3000

    print('Load 103k data')
    summary = []
    for batch_index in range(int(math.floor(train_size / batch_size))):
        print(batch_index)
        summary.append(load(batch_size=batch_size, batch_index=batch_index))

    dataset = list(itertools.chain.from_iterable(itertools.chain.from_iterable(summary)))

    print('Filter data')
    # Filter out mean, variance, min, max all zero
    dataset = list(filter(lambda l: np.linalg.norm(l[1:5]) > 0, dataset))

    dataset_asc_int = list(filter(lambda l: not l[6] and l[5] == 1, dataset))

    vectors = np.array([l[1:5] for l in dataset_asc_int])
    vectors /= np.linalg.norm(vectors, axis=1).reshape(-1, 1)
    labels = [l[0] for l in dataset_asc_int]

    print('Predict last 3k')
    answers = []
    for k, query in enumerate(vectors[-number_of_queries:]):
        answers.append(labels[np.dot(vectors[:-number_of_queries], query).argmax()])
        if k % 10 == 9:
            print('{}/{} - Accuracy: {}'.format(
                k + 1,
                number_of_queries,
                vector_accuracy(answers,
                                labels[len(labels) - number_of_queries:len(labels) - number_of_queries + k + 1])))

    print('Accuracy: {}'.format(vector_accuracy(answers, labels[-number_of_queries:])))


def vector_accuracy(a, b):
    assert len(a) == len(b)
    return sum(i == j for i, j in zip(a, b)) / len(a)


if __name__ == '__main__':
    falconn_test()
