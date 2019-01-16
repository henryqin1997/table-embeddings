import numpy as np
import falconn
import timeit
from numerical_space.load import load
import itertools
import json


def falconn_test():
    train_size = 100
    batch_size = 50
    batch_index = 0
    while batch_size * batch_index < train_size:
        print(batch_index)
        summary = load(batch_size=batch_size, batch_index=batch_index)
        dataset = np.array([l[1:-2] for l in list(itertools.chain.from_iterable(summary))])
        print(dataset)
        dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
        print(dataset)
        batch_index += 1


if __name__ == '__main__':
    falconn_test()
