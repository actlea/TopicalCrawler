#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: create_train_sample.py
@time: 16-4-23 上午11:58
"""
import os
import copy
import sys
import cPickle
import json
from lxml.html import builder as E
from lxml.html import tostring as htmltostring
from items import Document, UrlItem

from api import HtmlHelper
from util import FileHelper, ElementHelper, StringHelper, EncodeHelper
from cetd import get_page_link_ratio, get_aricle_cetd
from url import collect_urls, get_link_word_by_pair, get_link_word_pairs_by_wish, url_hashcode
from TopicalCrawl.classifier.multiclassifier import MultiClassifier

import re


def encode(content):
    try:
        if isinstance(content, unicode):
            content=content.encode('utf-8')
        return content
    except UnicodeDecodeError:
        return ''


def isdigit(s):
	for i in s:
		if i not in '0123456789\t\n \r':
			return True
	return False



def extract_all_links(infile, outfile, base_url=None, mode='a'):
    '''extract links as train and test samples'''
    content = FileHelper.readUTF8File(infile)
    if content is None:
        return None

    docstring = content.lower()
    doctree = HtmlHelper.create_doc(docstring, 'utf-8')

    if base_url is None:
        try:
            base_url = content.splitlines()[0]
        except IndexError or AttributeError:
            return None

    if doctree is None: return None

    doctree = HtmlHelper.pre_process_domtree(doctree)
    if doctree is None:
        return None
    url_items = []
    for item in get_link_word_by_pair(docstring, base_url, None): url_items.append(item)

    with open(outfile, mode) as fw:
        for item in url_items:
            anchor = item['anchor_text']
            url = item['url']
            tmp = anchor.decode('utf-8')
            print url, anchor
            if len(tmp)>5 and isdigit(tmp):
                fw.write('%s\t%s\n' %(url, anchor))



def html2words(docstring, base_url, encoding='utf-8', supervisior=None):
    """
    从网页源码中抽取正文
    :param docstring:
    :param encoding:
    :return:
    """
    string_size=sys.getsizeof(docstring)
    byte_size=string_size / (1024)
    if byte_size < 1:
        return

    docstring=docstring.lower()
    doctree=HtmlHelper.create_doc(docstring, encoding)

    if doctree is None: return None
    copy_doc=copy.deepcopy(doctree)
    try:

        link_ratio=get_page_link_ratio(copy_doc)
        print 'link_ratio: %f' % link_ratio

        if link_ratio > 0.6:
            print 'this is home page'
            # get all urls
            url_items = []
            for item in get_link_word_by_pair(docstring, base_url, supervisior): url_items.append(item)
            return None
    except ValueError:
        return None

    doctree=HtmlHelper.pre_process_domtree(doctree)
    if doctree is None:
        return None
    # get page title and para content
    para, title=HtmlHelper.get_article(doctree, debug=False)

    # get page meta keywords and meta description
    meta_description=HtmlHelper.get_meta_description(copy_doc)

    # get headlines in page
    cleaned_body=ElementHelper.get_body(doctree)
    headlines=HtmlHelper.get_headline_content_in_cleaned_body(cleaned_body)



    document=Document()
    document['base_url']=base_url
    document['title']=title
    document['meta']=meta_description
    document['headlines']=headlines
    document['para']=para
    # document['url_items']=url_items
    return document


def get_document(path):
    content = FileHelper.readUTF8File(path)
    if content is None:
        return None
    try:
        base_url = content.splitlines()[0]
    except IndexError or AttributeError:
        return None

    docstring = content.lower()
    doctree = HtmlHelper.create_doc(docstring, 'utf-8')

    if doctree is None: return None

    document = html2words(docstring, base_url)

    return document



if __name__=='__main__':
    infile='/mnt/UbutunShare/graduate/save3/d00001/'
    index=infile+'index'
    outfile='/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier/sample-data/'
    name='f00141'
    base_url='http://www.90oo.com/news/news-19756.html'
    pos_url='pos-url-anchor.txt'
    neg_url='neg-url-anchor.txt'

    extract_all_links(infile+name, outfile+neg_url, base_url=base_url, mode='a')