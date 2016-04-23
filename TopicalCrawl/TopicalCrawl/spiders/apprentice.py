#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
@author: actlea
@file: apprentice.py
@time: 16-3-19 下午2:28
@description:
"""

from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.http import Request


from TopicalCrawl.items import ResponseItem
from TopicalCrawl.url import *
from TopicalCrawl.util import Random
from TopicalCrawl.classifier.multiclassifier import load_classifier
from TopicalCrawl.link_relevance import *
from TopicalCrawl.parse import *







class Apprentice(CrawlSpider):
    name = 'apprentice'
    headers = {
        'Accept':'text/html',
        'User-Agent': Random.random_header()
    }
    allowed_domains=[
        # 'sports.sohu.com',
        # 'hupu.com',
        # 'sports.ifeng.com',
        # 'sports.qq.com',
        'sports.sina.com.cn'
    ]
    # start_urls = [
    #     'http://sports.sohu.com/',
    #     'http://www.hupu.com/',
    #     'http://sports.ifeng.com/',
    #     'http://sports.qq.com/',
    #     'http://sports.sina.com.cn/'
    # ]
    start_urls = ['http://sports.sina.com.cn/g/championsleague/']
    maxPageNumbers = 2000
    count = 0 #page has been download

    rules = (
        Rule(SgmlLinkExtractor(unique=True), callback='parse', follow=False),
    )

    level0 = {}
    level1 = {}
    graph = {}
    url_relevance = {}

    print '-----------load supervisior ----------------'
    model_path = '/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier'
    classifiers = load_classifier(model_path)


    def parse(self, response):
        response_item = ResponseItem()
        response_item['response'] = response
        response_item['count'] = self.count
        response_item['graph'] = self.graph
        response_item['level0'] = self.level0
        response_item['level1'] = self.level1
        response_item['url_relevance'] = self.url_relevance
        response_item['classifiers'] = self.classifiers
        yield  response_item

        self.count += 1
        if self.count > self.maxPageNumbers:
            return

        for url, priority in get_next_url(self.level0, self.level1,self.graph, self.url_relevance):
            priority = priority * 1000
            req = Request(url, callback=self.parse, headers=self.headers, priority=priority)

            # req.meta['classifiers'] = self.classifiers

            yield req


        # for urlitem in get_link_word_by_pair(response.body. response.url , response.encoding):
        #     url = urlitem['url']
        #     label, priority = urlitem['label'], urlitem['interestness']
        #     priority = priority*1000
        #
        #     if is_url_visited(url, visited_url_set) or int(label)==-1:
        #         continue
        #     req = Request(url, callback=self.parse, headers=self.headers, priority=priority)
        #     req.meta['link'] = urlitem
        #     yield req



if __name__ == '__main__':
    pass
