#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.tag import StanfordNERTagger

# os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk1.8.0_144/bin/java.exe'
jar = './stanford-ner-2018-02-27/stanford-ner.jar'
model = './stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz'
model_class = 3
tag_to_index = {'LOCATION': 0, 'PERSON': 1, 'ORGANIZATION': 2}

st = StanfordNERTagger(model, jar, encoding='utf-8')
