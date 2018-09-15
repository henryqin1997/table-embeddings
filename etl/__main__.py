#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy
from nltk.tag import StanfordNERTagger

data_dir = './data'

os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk1.8.0_144/bin/java.exe'
jar = './stanford-ner-2018-02-27/stanford-ner.jar'
model = './stanford-ner-2018-02-27/classifiers/english.muc.7class.distsim.crf.ser.gz'
model_class = 7
tag_to_index = {'LOCATION': 0, 'PERSON': 1, 'ORGANIZATION': 2, 'MONEY': 3, 'PERCENT': 4, 'DATE': 5, 'TIME': 6}

st = StanfordNERTagger(model, jar, encoding='utf-8')


def satisfy_variants(data):
    return data['tableType'] == 'RELATION' and data['hasHeader'] and data['headerRowIndex'] >= 0 and data[
        'tableOrientation'] in ['HORIZONTAL', 'VERTICAL']


class Table():
    def __init__(self, data):
        assert satisfy_variants(data)
        self.data = data
        self.header = self.extract_header(data)
        self.entities = self.extract_entities(data)
        self.attributes = numpy.transpose(self.entities)
        assert len(self.header) == len(self.attributes)

    @staticmethod
    def extract_header(data):
        if data['tableOrientation'] == 'HORIZONTAL':
            return numpy.array([item[data['headerRowIndex']] for item in data['relation']])
        else:
            return numpy.array(data['relation'][data['headerRowIndex']])

    @staticmethod
    def extract_entities(data):
        relation = data['relation']
        if data['tableOrientation'] == 'HORIZONTAL':
            return numpy.array([numpy.array([item[i] for item in relation]) for (i, val) in enumerate(relation[0]) if
                                i != data['headerRowIndex']])
        else:
            return numpy.array(
                [numpy.array(relation[i]) for (i, val) in enumerate(relation) if i != data['headerRowIndex']])

    def get_header(self):
        return self.header

    def get_entities(self):
        return self.entities

    def get_attributes(self):
        return self.attributes

    def get_data_md5(self):
        import hashlib
        return hashlib.md5(json.dumps(self.data).encode('utf-8')).hexdigest()

    def generate_ner_matrix(self):
        """
        Generate a 7*10 array.
        Each of the 7 rows corresponds to a NER tag.
        Each of the 10 columns correspond to a table attribute.
        Run Stanford NER on all values of each table attribute and mark the found tags as 1.
        """
        m = numpy.zeros((len(tag_to_index), 10))
        for i, attribute in enumerate(self.attributes):
            if i >= 10:
                break
            for value in attribute:
                print(value)
                ner_tags = st.tag(value.split())
                print(ner_tags)
                for ner_tag in ner_tags:
                    try:
                        m[tag_to_index[ner_tag[1]]][i] = 1
                    except KeyError:
                        pass
        return m

    def generate_wordlist_matrix(self, wordlist_to_index):
        """
        Given a wordlist, e.g. {'Date': 0, 'Name': 1, 'Opponent': 2},
        Generate a len(wordlist)*10 matrix.
        Each of the 10 columns correspond to a table attribute.
        If the label of this column is found in the wordlist, set 1 in that row.
        """
        m = numpy.zeros((len(wordlist_to_index), 10))
        for i, label in enumerate(self.header):
            if i >= 10:
                break
            try:
                m[wordlist_to_index[label]][i] = 1
            except KeyError:
                pass
        return m


def main():
    # print(st.tag('Michael Cafarella is doing a project on Tables Embedding in September 2018'.split()))
    with open(os.path.join(data_dir, 'sample'), encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if satisfy_variants(data) and len(data['relation']) in range(10, 20) and len(data['relation'][0]) in range(
                    10, 20):
                table = Table(data)
                if table.get_data_md5() == '0abc1124b6766c9bf281982a4e6adc5e':
                    json.dump(data,
                              open(os.path.join(data_dir, 'table-{}.json'.format(table.get_data_md5())), 'w+'),
                              indent=4)
                    print(table.get_header())
                    print(table.get_entities())
                    print(table.get_attributes())
                    print(table.generate_ner_matrix())
                    print(table.generate_wordlist_matrix({'Date': 0, 'Name': 1, 'Opponent': 2}))
                    print()


if __name__ == '__main__':
    main()
