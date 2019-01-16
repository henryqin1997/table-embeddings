import numpy as np
import falconn
import timeit
from numerical_space.load import load
import itertools
import json
import math


def falconn_test():
    train_size = 100
    batch_size = 50
    batch_index = 0
    number_of_queries = 10
    summary = [load(batch_size=batch_size, batch_index=batch_index) for batch_index in
               range(int(math.floor(train_size / batch_size)))]
    dataset = list(itertools.chain.from_iterable(itertools.chain.from_iterable(summary)))

    # Filter out mean, variance, min, max all zero
    dataset = list(filter(lambda l: np.linalg.norm(l[1:5]) > 0, dataset))

    dataset_asc_int = list(filter(lambda l: not l[6] and l[5] == 1, dataset))

    vectors = np.array([l[1:5] for l in dataset_asc_int])
    labels = np.array([l[0] for l in dataset_asc_int])
    print(vectors)
    print(labels)

    answers = []
    for query in vectors[-number_of_queries:]:
        print(np.dot(vectors[:-number_of_queries], query).argmax())
        answers.append(labels[np.dot(vectors[:-number_of_queries], query).argmax()])
    print(answers)


if __name__ == '__main__':
    falconn_test()
