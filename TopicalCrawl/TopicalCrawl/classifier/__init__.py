#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
@author: actlea
@file: __init__.py.py
@time: 16-3-17 上午9:43
@description:
"""

import os
from TopicalCrawl.classifier.multiclassifier import MultiClassifier
import json


def encode_m(txt):
    if txt is None:
        return ''
    if isinstance(txt, unicode):
        txt = txt.encode('utf-8', 'ignore')
    return txt

def get_document_class(name, model, class_file_name):
    src_dir='/mnt/UbutunShare/graduate/DataSet/scrapy_dataset/%s' % name

    supervisior=MultiClassifier(model)
    supervisior.load(path='/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier')
    json_dir=os.path.join(src_dir, 'json')

    list_file = os.listdir(json_dir)
    num = lambda x: int(x.split('.')[0])
    list_file.sort(cmp=lambda a,b:num(a)-num(b))

    document_class = {}
    dcoument_class_file=os.path.join(src_dir, class_file_name)

    for f in list_file:
        print '========================================'
        print 'parse file: %s ....' % f
        index = f.split('.')[0]

        document = json.load(open(json_dir+'/'+f))

        title = encode_m(document['title'])
        meta = encode_m(document['meta'])
        para = encode_m(document['div'])

        pure_text='\t'.join([title, meta, para])
        predicted_y, precision=supervisior.predict(pure_text)


        document_class[index]=predicted_y
    json.dump(document_class, open(dcoument_class_file, 'w'))


if __name__ == '__main__':
    names = ['ifeng','qq', 'qq_neg','other_neg','sina', 'sohu']
    for name in names:
        get_document_class(name, 'test-2-topic', 'document-3.class.json')