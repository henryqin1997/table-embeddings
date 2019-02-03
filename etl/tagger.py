#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.tag import StanfordNERTagger

# os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk1.8.0_144/bin/java.exe'
jar = './stanford-ner-2018-02-27/stanford-ner.jar'
model = './stanford-ner-2018-02-27/classifiers/english.muc.7class.distsim.crf.ser.gz'
model_class = 7
tag_to_index = {'LOCATION': 0, 'PERSON': 1, 'ORGANIZATION': 2, 'MONEY': 3, 'PERCENT': 4, 'DATE': 5, 'TIME': 6}

st = StanfordNERTagger(model, jar, encoding='utf-8')
