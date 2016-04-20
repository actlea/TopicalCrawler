#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: url.py
@time: 16-3-4 下午3:05
"""
from pybloom import ScalableBloomFilter
import urlparse
from hashlib import sha256, md5

from util import ElementHelper
from api import HtmlHelper
from config import *
from items import UrlItem
from api import WISH



unvisited_url_set = ScalableBloomFilter(mode=ScalableBloomFilter.SMALL_SET_GROWTH)
visited_url_set = ScalableBloomFilter(mode=ScalableBloomFilter.SMALL_SET_GROWTH)


#=======================================================
# base url function
def is_url_visited(url, set_name=visited_url_set):
    return set_name.add(url)

def get_scheme(url):
    if url is None:
        return None
    return urlparse.urlparse(url).scheme

def get_path(url):
    if url is None:
        return None
    return urlparse.urlparse(url).path

def get_domain(url):
    if url is None:
        return None
    return urlparse.urlparse(url).netloc

def set_abs_url(url, base_url=None):
    #if url[0]=='#' then url links to itself
    if url is None or url[0]=='#':
        return None
    return urlparse.urljoin(base_url, url)

def normalize_url(url, base_url=None):
    if get_domain(url) == '':
        url = set_abs_url(url, base_url)
    if get_path(url)=='':
        url += '/'
    if isinstance(url, unicode):
        url = url.encode('utf-8')
    return url


def m_strip(text):
    if text is None:
        return None
    return text.strip('\t\r\n \'\"')


def url_hashcode(url):
    code=None
    if type(url)==unicode:
        code = md5(url.encode('utf-8')).hexdigest()
    else:
        code = md5(url.__str__()).hexdigest()
    int_code = int(code, 16)
    return int_code

#========================================================
class Url(object):
    def __init__(self, url, base_url):
        self.url = normalize_url(url, base_url)

        self.path = get_path(self.url)
        self.domain = get_domain(self.url)
        self.scheme = get_scheme(self.url)

def should_collect_url(url, base_url):
    """ Determine whether the url should be collected based on the following criteria:
     1.位于同一个domain
     2.不爬去上层目录  http://sports.sina.com.cn/g/championsleague/  是 ·http://sports.sina.com.cn/·的下层目录
    """
    c = Url(url, base_url)
    p = Url(base_url, base_url)

    for word in ('tag', 'video'):
        if word in url:
            return False

    if c.scheme in ('http', 'https') and c.domain == p.domain and len(c.path)>1 : #and 'htm' in url
        return True
    return False


def neigb_text_of_link(atag, is_in_main_cluster=False, level=2):
    if not is_in_main_cluster:
        return ''

    cnt = 0
    negib_text = ''
    parent = atag.getparent()
    while cnt<level:
        parent_content = ElementHelper.element_text_tail(parent)
        if len(parent_content)>0:
            neigb_text = parent_content
            break
        else:
            cnt += 1
            parent = atag.getparent()
    return negib_text








def collect_urls(html, base_url, encoding=None):
    """ only collect url
    :param html: page string
    :param base_url:
    :param encoding:
    :return: list of url
    """
    h = HtmlHelper()
    doctree = h.create_doc(html, encoding)
    a_tags = ElementHelper.get_elements_by_tagnames(doctree, 'a')
    for a in a_tags:
        link = a.get('href',None)
        link = m_strip(link)
        if link is None or len(link)<2:continue
        if link[0]=='#': continue #link to itself
        link = normalize_url(link, base_url)

        #if url in non visited set
        if is_url_visited(link, unvisited_url_set):
            continue

        if not should_collect_url(link, base_url):
            continue
        #if url not in same domain

        yield link



def get_link_word_pairs_by_wish(docstring, base_url, supervisior=None, encoding='utf-8'):
    """ collect url from wish clusters, url_dict={url:, anchor_text:, neigb_text, interestness}
    :param doctree:
    :param base_url:
    :return: url pair dict
    """

    w = WISH()

    h = HtmlHelper()
    doctree = h.create_doc(docstring, encoding)
    if isinstance(base_url, unicode):
        base_url = base_url.encode('utf-8')
    clusters = w.get_cluster_records()




def get_link_word_by_pair(docstring, base_url, supervisior=None, encoding='utf-8'):
    """ collect urls from
    :param html:
    :param base_url:
    :return:
    """
    h = HtmlHelper()
    doctree = h.create_doc(docstring, encoding)
    if isinstance(base_url, unicode):
        base_url = base_url.encode('utf-8')
    a_tags = ElementHelper.get_elements_by_tagnames(doctree, 'a')

    for a in a_tags:

        link = a.get('href',None)

        link = a.get('href',None)
        link = m_strip(link)

        if link is None or len(link)<2:continue
        if link[0]=='#': continue #link to itself
        link = normalize_url(link, base_url)

        #if url in non visited set
        if is_url_visited(link, unvisited_url_set):
            continue

        # if not should_collect_url(link, base_url):
        #     continue


        link_item = UrlItem()
        link_item['parent_url'] = base_url
        link_item['url'] = link
        link_item['anchor_text'] = ElementHelper.element_text_content(a).encode('utf-8')
        link_item['neigb_text'] = ''
        if supervisior is not None:
            link_item['label'], link_item['interestness'] = supervisior.predict(link_item['anchor_text'])
        else:
            link_item['label'], link_item['interestness'] = '1', 0.0  #1为负样本

        yield link_item










if __name__ == '__main__':
    pass
