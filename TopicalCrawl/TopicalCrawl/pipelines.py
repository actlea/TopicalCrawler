# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


from parse import *





class TopicalcrawlPipeline(object):
    def process_item(self, response_item, spider):
        response = response_item['response']
        count = response_item['count']
        graph = response_item['graph']
        level0 = response_item['level0']
        level1 = response_item['level1']
        url_relevance = response_item['url_relevance']
        classifiers = response_item['classifiers']

        parse_page(response_item, count)

        return response_item
