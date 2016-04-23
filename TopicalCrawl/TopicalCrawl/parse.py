#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: parse.py
@time: 16-4-23 下午10:32
"""


from htmlParse import *
from cetd import get_page_link_ratio

import copy


def is_hub_page(doc):
    try:
        link_ratio=get_page_link_ratio(doc)
        print 'link_ratio: %f' % link_ratio

        if link_ratio > 0.6:
            return True
        else:
            return False
    except ValueError:
        return False


def get_valid_page(page, encoding):
    try:
        content = EncodeHelper.get_unicode(page, encoding)
        if content is None:
            return None
        content = content.encode('utf-8')
        return content
    except UnicodeError:
        return None


def parse_tag_structure(doc):
    tmpdoc = copy.deepcopy(doc)
    doctree = HtmlHelper.pre_process_domtree(tmpdoc)


def parse_page(response_item, save_dir):
    response = response_item['response']
    count = response_item['count']
    graph = response_item['graph']
    level0 = response_item['level0']
    level1 = response_item['level1']
    url_relevance = response_item['url_relevance']
    classifiers = response_item['classifiers']

    #查看是否为hub网页
    url = response.url
    body = response.body
    encoding = response.encoding

    valid_page = get_valid_page(body, encoding)

    if valid_page is None:
        return

    #解析网页的标签结构
    docstring = valid_page.lower()
    doctree = HtmlHelper.create_doc(valid_page, encoding)

    if is_hub_page(copy.deepcopy(doctree)):
        #如果是hub网页，就仅抽取链接
        pass
    else:

