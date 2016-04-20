#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
@author: actlea
@file: test.py
@time: 16-3-29 下午4:11
@description:
"""

from TopicalCrawl.classifier.multiclassifier import MultiClassifier
from items import Document, UrlItem
import os
import copy
import sys
import cPickle
import json

from api import HtmlHelper
from util import FileHelper, ElementHelper, StringHelper
from cetd import get_page_link_ratio, get_aricle_cetd

from TopicalCrawl.classifier.base import  TestResult

from htmlParse import *
import numpy


#============================================================
#测试基于标签的网页分类

def get_document(name):
    home = '/mnt/UbutunShare/graduate/DataSet/PageClassification/%s/' %name

    json_dir = home + 'json/'

    mkdirs(home, json_dir)

    history = home+'history.txt'
    visited_files = get_history(history)
    finput = open(history, 'w')


    yule = home+'yule/'
    sport = home +'sport/'
    finance = home +'finance/'
    junshi = home + 'junshi/'


    for dir in [yule, sport, finance, junshi]:
        print dir
        for f in os.listdir(dir):
            if f in visited_files:
                continue
            print '========================================'
            print 'parse file: %s ....' % f

            t = dir + f
            content=FileHelper.readUTF8File(t)
            if content is None:
                continue
            try:
                base_url=content.splitlines()[0]
            except IndexError or AttributeError:
                continue


            document=html2words(content, base_url, encoding='utf-8')
            # if document is None:
            #     os.remove(t)
            #     continue
            # div = document['para']
            #
            # if len(div)<50:
            #     os.remove(t)
            #     continue

            json_file=os.path.join(json_dir, f.split('.')[0] + '.json')
            save(document, json_file)
            finput.writelines(f + '\n')
    finput.close()

def encode_m(text):
    if text is None:
        return ''
    if isinstance(text, unicode):
        text = text.encode('utf-8')
    return text


def predict_t_d_m(name, model):
    home = '/mnt/UbutunShare/graduate/DataSet/PageClassification/%s/' %name

    json_dir = home + 'json/'
    yule = home+'yule/'
    sport = home +'sport/'
    finance = home +'finance/'
    junshi = home + 'junshi/'
    temp = home+'temp/'
    mkdirs(temp)


    supervisior=MultiClassifier(model)
    supervisior.load(path='/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier')

    tmpf0 = open(temp+'0.txt', 'w')
    tmpf1 = open(temp+'1.txt', 'w')
    tmpf2 = open(temp+'2.txt', 'w')
    tmpf3 = open(temp+'3.txt', 'w')
    tmpf4 = open(temp+'4.txt', 'w')

    document_class = {'0':{}, '1':{}, '2':{}, '3':{}, '4':{}}


    json_files = os.listdir(json_dir)
    for _file in json_files:
        print '=============================='
        print 'file: %s' %_file


        document = json.load(open(json_dir+_file, 'r'))
        title = encode(document['title'])
        meta = encode(document['meta'])
        div = encode(document['div'])

        content = div
        predicted_y, precision=supervisior.predict(content)
        tmpf0.write('%s\t%s\n' %(predicted_y, _file))
        document_class['0'][_file] = predicted_y

        content = meta
        predicted_y, precision=supervisior.predict(content)
        tmpf1.write('%s\t%s\n' %(predicted_y, _file))
        document_class['1'][_file] = predicted_y

        content = ' '.join([title, meta, div])
        predicted_y, precision=supervisior.predict(content)
        tmpf2.write('%s\t%s\n' %(predicted_y, _file))
        document_class['2'][_file] = predicted_y

        content = ' '.join([title, meta, meta,  div])
        predicted_y, precision=supervisior.predict(content)
        tmpf3.write('%s\t%s\n' %(predicted_y, _file))
        document_class['3'][_file] = predicted_y

        content = ' '.join([title, meta,meta, meta, div])
        predicted_y, precision=supervisior.predict(content)
        tmpf4.write('%s\t%s\n' %(predicted_y, _file))
        document_class['4'][_file] = predicted_y

    json.dump(document_class, open(temp+'test_result.txt', 'w'))
    tmpf0.close()
    tmpf1.close()
    tmpf2.close()
    tmpf3.close()
    tmpf4.close()


def test_result(name):
    home = '/mnt/UbutunShare/graduate/DataSet/PageClassification/%s/' %name
    temp = home+'temp/'

    document_labels = json.load(open(temp+'test_result.txt'))

    exp=lambda x:x.split('.')[0]
    em = lambda x:x.encode('utf-8')

    true_y = {}
    with open(temp+'true_y.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            label, _file = line.split('\t')
            true_y[exp(_file)] = label
    file2id = {_file:_id for _id, _file in enumerate(true_y)}

    tmp0_y, tmp1_y, tmp2_y, tmp3_y, tmp4_y = document_labels['0'], \
                                             document_labels['1'], document_labels['2'],\
                                             document_labels['3'], document_labels['4']

    tmp1_y = {em(exp(_file)):_label.encode('utf-8') for _file, _label in tmp1_y.items()}
    tmp2_y = {em(exp(_file)):_label.encode('utf-8') for _file, _label in tmp2_y.items()}
    tmp3_y = {em(exp(_file)):_label.encode('utf-8') for _file, _label in tmp3_y.items()}
    tmp0_y = {em(exp(_file)):_label.encode('utf-8') for _file, _label in tmp0_y.items()}
    tmp4_y = {em(exp(_file)):_label.encode('utf-8') for _file, _label in tmp4_y.items()}

    tmp0_array = numpy.zeros(len(true_y))
    tmp4_array = numpy.zeros(len(true_y))
    tmp1_array = numpy.zeros(len(true_y))
    tmp2_array = numpy.zeros(len(true_y))
    tmp3_array = numpy.zeros(len(true_y))
    _true_y =  numpy.zeros(len(true_y))

    for _file in true_y:
        _id = file2id[_file]
        try:
            tmp0_array[_id] = tmp0_y[_file]
            tmp4_array[_id] = tmp4_y[_file]
            tmp1_array[_id] = tmp1_y[_file]
            tmp2_array[_id] = tmp2_y[_file]
            tmp3_array[_id] = tmp3_y[_file]
            _true_y[_id] = true_y[_file]
        except KeyError:
            continue
    tmp1_result = TestResult(_true_y, tmp1_array, None)
    tmp2_result = TestResult(_true_y, tmp2_array, None)
    tmp3_result = TestResult(_true_y, tmp3_array, None)
    tmp0_result = TestResult(_true_y, tmp0_array, None)
    tmp4_result = TestResult(_true_y, tmp4_array, None)

    print '============tmp0=============='
    print tmp0_result.accuracy_labels
    print tmp0_result.recall_labels
    print tmp0_result.accuracy_score
    tmp0_result.show_result()

    print '============tmp1=============='
    print tmp1_result.accuracy_labels
    print tmp1_result.recall_labels
    print tmp1_result.accuracy_score
    tmp1_result.show_result()
    print '============tmp2=============='
    print tmp2_result.accuracy_labels
    print tmp2_result.recall_labels
    print tmp2_result.accuracy_score
    tmp2_result.show_result()
    print '============tmp3=============='
    print tmp3_result.accuracy_labels
    print tmp3_result.recall_labels
    print tmp3_result.accuracy_score
    tmp3_result.show_result()


    print '============tmp4=============='
    print tmp4_result.accuracy_labels
    print tmp4_result.recall_labels
    print tmp4_result.accuracy_score
    tmp4_result.show_result()

if __name__=='__main__':
    name = 'Test4'
    # get_document('Test4')
    # predict_t_d_m(name, 'test-3-topic')
    test_result(name)






