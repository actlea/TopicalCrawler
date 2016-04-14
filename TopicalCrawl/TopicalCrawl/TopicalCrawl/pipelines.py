# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

from htmlParse import *







class TopicalcrawlPipeline(object):
    def process_item(self, item, spider):
        response = item['response']
        count = item['count']
        if response.meta.has_key('supervisior'):
            supervisior6, supervisior7 = response.meta.get('supervisior')

            save_response('Test7', response, count, supervisior6, supervisior7)

            return item
