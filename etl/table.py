#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import json
import hashlib


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

    def get_data(self):
        return self.data

    def get_header(self):
        return self.header

    def get_entities(self):
        return self.entities

    def get_attributes(self):
        return self.attributes

    def get_data_md5(self):
        return hashlib.md5(json.dumps(self.data).encode('utf-8')).hexdigest()

    def generate_ner_matrix(self, tagger, tag_to_index):
        """
        Generate a 7*10 array.
        Each of the 7 rows corresponds to a NER tag.
        Each of the 10 columns corresponds to a table attribute.
        Run Stanford NER on all values of each table attribute and mark the found tags as 1.
        """
        ner_tags_dict = {}
        m = numpy.zeros((len(tag_to_index), 10))
        for i, attribute in enumerate(self.attributes):
            if i >= 10:
                break
            for value in attribute:
                try:
                    ner_tags = ner_tags_dict[repr(value.split())]
                except KeyError:
                    ner_tags = tagger.tag(value.split())
                    ner_tags_dict[repr(value.split())] = ner_tags
                for ner_tag in ner_tags:
                    try:
                        m[tag_to_index[ner_tag[1]]][i] = 1
                    except KeyError:
                        pass
        return m

    def generate_wordlist_matrix(self, wordlist_to_index):
        """
        Given a wordlist, e.g. {'Date': 0, 'Name': 1, 'Opponent': 2},
        Generate a (len(wordlist)+1)*10 matrix.
        Each of the first len(wordlist) rows corresponds to a word in wordlist.
        Each of the 10 columns corresponds to a table attribute.
        If the label of this column is found in the wordlist, set 1 in that row.
        If the label is not found in the wordlist, set 1 in the last row.
        """
        m = numpy.zeros((len(wordlist_to_index) + 1, 10))
        for i, label in enumerate(self.header):
            if i >= 10:
                break
            try:
                m[wordlist_to_index[label.lower().strip()]][i] = 1
            except KeyError:
                m[-1][i] = 1
                pass
        return m
