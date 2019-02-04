import sys


def value_to_features(value):
    if value < 0:
        return 'NULL'
    features = {}
    bools = list(map(int, list('{0:b}'.format(int(value)).zfill(11))))
    features['TEXT'] = bools[-1]
    features['SYMBOL'] = bools[-2]
    features['NUMBER'] = bools[-3]
    features['LOCATION'] = bools[-4]
    features['PERSON'] = bools[-5]
    features['ORGANIZATION'] = bools[-6]
    features['DATE'] = bools[-7]
    features['IS_NUMERIC'] = bools[-8]
    features['IS_FLOAT'] = bools[-9]
    if bools[-10] == 0 and bools[-11] == 0:
        features['IS_ORDERED'] = 'rand'
    elif bools[-10] == 1 and bools[-11] == 0:
        features['IS_ORDERED'] = 'desc'
    elif bools[-10] == 0 and bools[-11] == 1:
        features['IS_ORDERED'] = 'asc'
    else:
        features['IS_ORDERED'] = 'error'
    return features


if __name__ == '__main__':
    value = sys.argv[1]
    print(value_to_features(int(value)))
