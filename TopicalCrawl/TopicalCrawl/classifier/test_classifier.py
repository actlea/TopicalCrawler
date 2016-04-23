#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: test_classifier.py
@time: 16-4-16 下午9:09
"""

import re
import os
from api import *
from base import *
from util import  EncodeHelper
from biclassifier import BiClassifier


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


def read_page(path):
    with open(path) as fr:
        content = fr.read()
        url = content.splitlines()[0]
        page = '\n'.join(content.splitlines()[1:])
    return url, page


def make_density(content):
    if content is None:
        return ''
    if isinstance(content, str):
        content = content.decode('utf-8')
    content = re.sub(r'\t', '', content)
    content = re.sub(r'[\r,\n]\s', '', content)
    content = content.replace('&nbsp', '')
    content = ''.join([x.strip() for x in content if len(x.strip()) != 0])
    if isinstance(content, unicode):
        content = content.encode('utf-8', 'ignore')

    return content



#过滤非体育类型的网页
def bi_filter(textModel, urlModel, url,  raw_page):
    url_pred, url_prob = urlModel.predict(url)
    text_pred, text_prob = textModel.predict(raw_page)

    if url_pred==text_pred:
        return url_pred
    elif url_prob>text_prob and text_prob<0.9:
        return url_pred
    else:
        return text_pred


def test_bi_classifier(X1, X2, Y):
    n_samples = len(Y)
    c1 = BiClassifier('test-9-text', tokenize='url', feature_select=False) #url model
    c2 = BiClassifier('test-1-text', feature_select=True) #text model
    c1.load()
    c2.load()
    # Y1 = c1.predicts(X1)
    # print 'url_only'
    # BiClassifier.show_result(Y1, Y)
    # Y2 = c2.predicts(X2)
    # print 'text only'
    # BiClassifier.show_result(Y2, Y)
    Y3 = [bi_filter(c2, c1, X1[i], X2[i]) for i in range(n_samples)]

    print 'url and text'
    BiClassifier.show_result(Y3, Y)

def get_rawpage(page):
    page = EncodeHelper.get_unicode(page)
    if page is None: return None
    cleaned_page = HtmlHelper.get_cleaned_html(page)
    raw_page = remove_tags(cleaned_page)

    raw_page = make_density(raw_page)
    return raw_page


def stripNonAlphaNum(text):
    text = text.decode('utf-8')
    text=re.sub ( r'[^a-zA-Z0-9]', ' ', text )
    return text.encode ( 'utf-8' )



if __name__=='__main__':
    test_dir = '/mnt/UbutunShare/graduate/DataSet/PageClassification/Test5/'
    name = 'finance/'
    name = 'junshi/'
    names = ['finance/', 'junshi/', 'yule/', 'sport/']





