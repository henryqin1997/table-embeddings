'''
To load all numerical columns from tables.
'''
import numpy as np


def summary(label, column):
    """
    Input: a column of numerical values in list version
    Output: a list of label, mean, variance, min, max, is_ordered, is_real
    """
    column = np.array(column)
    values = column.astype(np.float)
    return [label, np.mean(values), np.var(values), np.min(values), np.max(values),
            1 if np.all(np.diff(values) > 0) else -1 if np.all(np.diff(values) < 0) else 0,
            '.' in ''.join(column)]


def load():
    """
    return summarized data for a table, ie. a list of points summary.
    """


if __name__ == '__main__':
    print(summary('test', ['5']))
