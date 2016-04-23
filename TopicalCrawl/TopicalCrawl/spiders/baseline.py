#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: baseline.py
@time: 16-3-3 下午3:11
"""
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.http import Request



from TopicalCrawl.items import ResponseItem
from TopicalCrawl.url import *
from TopicalCrawl.util import Random


from TopicalCrawl.config import  *

from TopicalCrawl.classifier.multiclassifier import MultiClassifier

supervisior7=MultiClassifier('test-7-topic')
supervisior7.load(path='/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier')

supervisior6=MultiClassifier('test-6-topic')
supervisior6.load(path='/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier')


'''
class Downloader(CrawlSpider):
    name = 'download'
    headers = {
        'Accept':'text/html',
        'User-Agent': Random.random_header()
    }
    allowed_domains = []
    # allowed_domains=[
    #     'auto.sina.com.cn',
    #     'www.autohome.com.cn'
    #     # 'health.people.com.cn',
    #     # 'health.sohu.com'
    #     # 'mil.news.sina.com.cn',
    #     # 'mil.huanqiu.com'
    #     # 'finance.sina.com.cn'
    #     # 'sports.sohu.com',
    #     # 'hupu.com',
    #     # 'sports.ifeng.com',
    #     # 'sports.qq.com',
    #     # 'news.qq.com'
    # ]
    # start_urls = [
    #     'http://sports.sohu.com/',
    #     'http://www.hupu.com/',
    #     'http://sports.ifeng.com/',
    #     'http://sports.qq.com/',
    #     'http://sports.sina.com.cn/'
    # ]
    start_urls = Test7_URL
        # 'http://auto.sina.com.cn/',
        # 'http://www.autohome.com.cn/beijing/'
        # 'http://health.people.com.cn/',
        # 'http://health.sohu.com/'
        # 'http://mil.news.sina.com.cn/',
        # 'http://mil.huanqiu.com/'
        # 'http://finance.sina.com.cn/'
    count = 85


    maxPageNumbers = 50000

    rules = (
        Rule(SgmlLinkExtractor(unique=True), callback='parse', follow=False),
    )

    print 'download start--------------'

    def parse(self, response):
        item = ResponseItem()
        item['response'] = response
        item['count'] = self.count

        self.count += 1
        if self.count > self.maxPageNumbers:
            return

        yield  item

        print '===================================='
        print 'parse url!'
        # for url in collect_urls(response.body, response.url, response.encoding):
        for item in get_link_word_by_pair(response.body, response.url, encoding=response.encoding):
            url = item['url']
            if is_url_visited(url, visited_url_set):
                continue
            # print 'url:%s' %url
            req = Request(url, callback=self.parse, headers=self.headers)
            req.meta['link'] = item
            req.meta['supervisior'] = [supervisior6, supervisior7]
            yield req




if __name__ == '__main__':
    pass
'''