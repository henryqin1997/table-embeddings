import json
import os
import numpy
import locale
from etl import Table
from numerical_space.load import is_numeric, nst_encoding

training_data_dir = 'data/train'
training_files_json = 'data/training_files_shuffle.json'
training_files = json.load(open(training_files_json))
training_data_100_sample_dir = 'data/train_100_sample'
training_files_100_sample_json = 'data/training_files_100_sample.json'
training_files_100_sample = json.load(open(training_files_100_sample_json))
training_data_domain_sample_dir = 'data/domain_samples/googlecom'
training_files_domain_sample_json = 'data/domains/googlecom.json'
training_files_domain_sample = json.load(open(training_files_domain_sample_json))
training_data_domain_schemas_dir = 'data/domain_schema_files'
training_files_domain_schemas_json = 'data/domain_schema_files_dict.json'
training_files_domain_schemas = list(json.load(open(training_files_domain_schemas_json)).values())

tag_to_index = {'LOCATION': 0, 'PERSON': 1, 'ORGANIZATION': 2}


def one_hot(row):
    assert len(row) > 0
    sum = row_sum(row)
    row_converted = numpy.zeros(2 ** len(row))
    assert sum < len(row_converted)
    row_converted[sum] = 1
    return row_converted


def row_sum(row):
    return int(round(sum(numpy.array([(2 ** i) * num for (i, num) in enumerate(row)]))))


def load_data(batch_size, batch_index=0, training_data_dir=training_data_dir, training_files=training_files):
    # load training data from file, to be implemented
    # put size number of data into one array
    # start from batch_index batch
    # return two arrays: input, target
    batch_files = training_files[batch_size * batch_index:batch_size * (batch_index + 1)]
    batch_files_ner = list(map(lambda batch_file: batch_file.rstrip('.json') + '_ner.csv', batch_files))
    batch_files_nst = list(map(lambda batch_file: batch_file.rstrip('.json') + '_nst.csv', batch_files))
    batch_files_date = list(map(lambda batch_file: batch_file.rstrip('.json') + '_date.csv', batch_files))
    batch_files_wordlist = list(map(lambda batch_file: batch_file.rstrip('.json') + '_wordlist.csv', batch_files))
    ner_inputs = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_ner), delimiter=',') for batch_file_ner in
         batch_files_ner])
    nst_inputs = numpy.array(
        [list(map(to_int, numpy.genfromtxt(os.path.join(training_data_dir, batch_file_nst), delimiter=',')[0])) for
         batch_file_nst in batch_files_nst])
    date_inputs = numpy.array(
        [list(map(to_int, numpy.genfromtxt(os.path.join(training_data_dir, batch_file_date), delimiter=','))) for
         batch_file_date in batch_files_date])
    targets = numpy.array(
        [numpy.genfromtxt(os.path.join(training_data_dir, batch_file_wordlist), delimiter=',') for batch_file_wordlist
         in batch_files_wordlist])

    inputs_transformed = []
    targets_transformed = []

    assert len(ner_inputs) == len(nst_inputs)
    assert len(ner_inputs) == len(date_inputs)
    for i in range(len(ner_inputs)):
        # print(batch_files[i])
        table = Table(json.load(open(os.path.join(training_data_dir, batch_files[i]))))
        column_num = len(table.get_header())
        attributes = table.get_attributes()
        ner_input = ner_inputs[i]
        nst_input = nst_inputs[i]
        date_input = date_inputs[i]
        target = targets[i]
        assert len(ner_input) == len(tag_to_index)

        # Encode 3 class NER (4:location, 5:person, 6:organization)
        new_input_transformed = numpy.array([int(round(sum([(2 ** (i + 3)) * num for (i, num) in enumerate(ner_row)])))
                                             if idx < column_num else -1 for idx, ner_row in
                                             enumerate(ner_input.transpose())]).transpose()
        # print('ner', new_input_transformed)
        # Add encoded NST and date (1:text, 2:symbol, 3:number, 7:date)
        new_input_transformed = new_input_transformed + numpy.array(nst_input) + numpy.array(date_input) * (2 ** 6)
        # print('nst', numpy.array(nst_input))
        # print('date', numpy.array(date_input) * (2 ** 6))

        # Check is_numeric, is_float and is_ordered (8:is_numeric, 9:is_float, 10:is_ordered)
        is_numeric_input = [-1] * 10
        is_float_input = [-1] * 10
        is_ordered_input = [-1] * 10
        for idx in range(min(column_num, 10)):
            is_numeric_input[idx] = 0
            is_float_input[idx] = 0
            is_ordered_input[idx] = 0
            if nst_input[idx] == nst_encoding([True, False, False]) or \
                    nst_input[idx] == nst_encoding([True, True, False]):
                attribute = attributes[idx]
                if all(list(map(lambda n: is_numeric(n) or n.upper() in ['', 'NA', 'N/A'], attribute))):
                    is_numeric_input[idx] = 1
                    is_float_input[idx] = int('.' in ''.join(attribute))
                    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
                    values = numpy.array(list(map(locale.atof, filter(is_numeric, attribute))))
                    # 0: random, 1: desc, 2: asc
                    is_ordered_input[idx] = 2 if numpy.all(numpy.diff(values) > 0) else \
                        1 if numpy.all(numpy.diff(values) < 0) else 0

        new_input_transformed = new_input_transformed + \
                                numpy.array(is_numeric_input) * (2 ** 7) + \
                                numpy.array(is_float_input) * (2 ** 8) + \
                                numpy.array(is_ordered_input) * (2 ** 9)
        # print('is_numeric', numpy.array(is_numeric_input) * (2 ** 7))
        # print('is_float', numpy.array(is_float_input) * (2 ** 8))
        # print('is_ordered', numpy.array(is_ordered_input) * (2 ** 9))

        # Change all negative values to -1 (empty column)
        new_input_transformed = numpy.array([x if x >= 0 else -1 for x in new_input_transformed])
        # print('overall', new_input_transformed)

        inputs_transformed.append(new_input_transformed)

        targets_transformed.append(
            numpy.array([index_of(list(map(lambda num: int(round(num)), row)), 1) if idx < column_num else -1 for
                         idx, row in enumerate(target.transpose())]).transpose())
    return numpy.array(inputs_transformed), numpy.array(targets_transformed)


def load_data_100_sample(batch_size, batch_index=0):
    return load_data(batch_size, batch_index=batch_index,
                     training_data_dir=training_data_100_sample_dir,
                     training_files=training_files_100_sample)


def load_data_domain_sample(batch_size, batch_index=0):
    return load_data(batch_size, batch_index=batch_index,
                     training_data_dir=training_data_domain_sample_dir,
                     training_files=training_files_domain_sample)


def load_data_domain_schemas(batch_size, batch_index=0):
    return load_data(batch_size, batch_index=batch_index,
                     training_data_dir=training_data_domain_schemas_dir,
                     training_files=training_files_domain_schemas)


def index_of(l, n):
    try:
        return list(l).index(n)
    except ValueError:
        return -1


def to_int(n):
    return int(round(n))


if __name__ == '__main__':
    print(load_data_domain_sample(10, batch_index=0))
