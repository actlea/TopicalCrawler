#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: spider.py
@time: 16-3-2 下午7:14
"""

import scrapy
from scrapy.contrib.spiders import CrawlSpider
from scrapy.http import Request

from ..items import ResponseItem

class TopicalCrawler(CrawlSpider):
    name = 'spider'
    # start_urls = ['http://www.hupu.com/',
    #               'http://sports.sina.com.cn/',
    #               'http://sports.sina.com.cn/g/championsleague/',
    #               'http://sports.qq.com/',
    #               'http://sports.sohu.com/',
    #               'http://sports.ifeng.com/',
    #               'http://match.sports.sina.com.cn/index.html',
    #               'http://live.sports.ifeng.com/index.shtml'
    # ]
    start_url = ['http://sports.sina.com.cn/']

    def parse(self, response):
        item = ResponseItem()
        item['response'] = response
        yield item

        for link in self.start_urls:
            yield Request(link, callback=self.parse)






if __name__ == '__main__':
    pass
