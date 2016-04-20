# #!/usr/bin/env python
# #-*- coding: utf-8 -*-
#
# """
# @author: actlea
# @file: url_online_train.py
# @time: 16-3-21 下午5:11
# @description:利用url分类器进行在线训练的爬虫
# """
#
# from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
# from scrapy.contrib.spiders import CrawlSpider, Rule
# from scrapy.http import Request
#
#
# from TopicalCrawl.items import ResponseItem
# from TopicalCrawl.url import *
# from TopicalCrawl.util import Random
# from TopicalCrawl.classifier.multiclassifier import MultiClassifier
# from TopicalCrawl.classifier.pure_url_classify import UrlClassifier
#
# from TopicalCrawl.htmlParse import html2words
#
#
#
#
#
# class UrlOnlineTrain(CrawlSpider):
#     name = 'url_online_train'
#     headers = {
#         'Accept':'text/html',
#         'User-Agent': Random.random_header()
#     }
#     allowed_domains=[
#         # 'sports.sohu.com',
#         # 'hupu.com',
#         # 'sports.ifeng.com',
#         # 'sports.qq.com',
#         'sports.sina.com.cn'
#     ]
#     # start_urls = [
#     #     'http://sports.sohu.com/',
#     #     'http://www.hupu.com/',
#     #     'http://sports.ifeng.com/',
#     #     'http://sports.qq.com/',
#     #     'http://sports.sina.com.cn/'
#     # ]
#     start_urls = ['http://sports.sina.com.cn/g/championsleague/']
#     maxPageNumbers = 2000
#     count = 0 #page has been download
#
#     rules = (
#         Rule(SgmlLinkExtractor(unique=True), callback='parse', follow=False),
#     )
#
#     print '-----------load supervisior ----------------'
#     supervisior = MultiClassifier('test-zh-logreg')
#     model_path = '/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier'
#     supervisior.load(path=model_path)
#
#     url_supervisior = UrlClassifier('test-zh-url')
#     url_supervisior.load(path=model_path)
#
#
#     def parse(self, response):
#         item = ResponseItem()
#         item['response'] = response
#         item['count'] = self.count
#         yield  item
#
#         self.count += 1
#         if self.count > self.maxPageNumbers:
#             return
#
#
#         for urlitem in get_link_word_by_pair(response.body. response.url , response.encoding):
#             url = urlitem['url']
#             label, priority = urlitem['label'], urlitem['interestness']
#             priority = priority*1000
#
#             if is_url_visited(url, visited_url_set) or int(label)==-1:
#                 continue
#             req = Request(url, callback=self.parse, headers=self.headers, priority=priority)
#             req.meta['link'] = urlitem
#             yield req
#
